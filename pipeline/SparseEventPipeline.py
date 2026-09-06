"""Sparse event localisation pipeline: extract, review, generate questions."""

import json
from pathlib import Path

from pipeline.AIParser import AIParser
from pipeline.ConfigLoader import ConfigLoader
from pipeline.TemporalPipeline import parse_json_response
from pipeline.frame_parser import FrameParser
from pipeline.image_preprocessor import ImagePreprocessor
from pipeline.timeframe_converter import TimeframeConverter


def _frame_sort_key(event: dict):
    """Sort key ordering events by their numeric frame timestamp.

    Frame filenames are epoch timestamps, so the numeric stem gives global
    chronological order. Events without a parseable frame go last.
    """
    # Numeric frame names preserve order; invalid frames sort last.
    try:
        return (0, int(Path(str(event.get("frame", ""))).stem))
    except ValueError:
        return (1, 0)


class SparseEventPipeline:
    """
    Pipeline to assess sparse event localisation memory.

    Stages:
        extract   - split frames into sections and detect noteworthy events
                    with the VLM
        review    - sort detections by frame timestamp, merge duplicates,
                    filter spurious detections with the LLM, and write the
                    human-reviewable events.json
        questions - generate sparse-event questions from the
                    human-verified events.json
    """

    STAGES = ("extract", "review", "questions")

    def __init__(
        self,
        config_path: str | Path,
        prompt: str,
        filter_prompt: str,
        question_prompt: str,
    ):
        # Store prompts; helpers are created when config loads.
        self.config_path = config_path
        self.prompt = prompt
        self.filter_prompt = filter_prompt
        self.question_prompt = question_prompt
        self.config = None
        self.frame_parser = None
        self.ai_parser = None
        self.frame_converter = None

    def load_config(self) -> dict:
        """
        Load config from ConfigLoader and create instances of
        helper classes
        """
        # Load shared pipeline settings.
        self.config = ConfigLoader(self.config_path).load()

        # Configure frame sectioning from the loaded settings.
        self.frame_parser = FrameParser(
            frames_dir=self.config["frames_dir"],
            output_dir=self.config["sections_dir"],
            frames_per_section=self.config["frames_per_section"],
            step_size=self.config["step"],
            move=self.config.get("move", False),
            preprocessor=ImagePreprocessor.from_config(self.config),
        )

        # Convert frame filenames to elapsed seconds from real timestamps.
        self.frame_converter = TimeframeConverter(self.config["frames_dir"])

        # Initialize the configured local or API backend.
        self.ai_parser = AIParser(config=self.config)
        return self.config

    def run(self, stage: str | None = None) -> None:
        """
        Run one sparse-event pipeline stage. When stage is None, run extract
        followed by review. The questions stage must be run separately,
        after a human has reviewed events.json.
        """
        # Validate the requested stage before loading models.
        if stage is not None and stage not in self.STAGES:
            raise ValueError(f"stage must be one of {self.STAGES}")

        # Prepare frame and model helpers for the selected stages.
        self.load_config()

        if stage is None or stage == "extract":
            # call VLM to extract section of files.
            self._extract()
        if stage is None or stage == "review":
            # call LLM to review and filter detections.
            self._review()
        if stage == "questions":
            self._generate_questions()

    def _load_section_manifest(self) -> dict:
        """Load the per-section seconds manifest written by FrameParser."""
        manifest_path = Path(self.config["sections_dir"]) / "sections.json"
        if not manifest_path.is_file():
            print(f"  Warning: section manifest not found at {manifest_path}")
            return {}
        with open(manifest_path) as manifest_file:
            return json.load(manifest_file)

    def _frame_seconds(self, frame: str | None) -> float | None:
        """Convert one frame filename to elapsed seconds from the video start."""
        if not frame:
            return None
        try:
            return self.frame_converter.frame_to_seconds(frame)
        except ValueError:
            return None

    def _events_with_seconds(self, section_result: dict, manifest: dict) -> list[dict]:
        """Parse a section's VLM response into events carrying section seconds.

        Section seconds come from the section manifest written at
        frame-parsing time, so the model never has to produce the
        timestamps itself.
        """
        section_name = section_result.get("section")
        section_info = (manifest.get("sections") or {}).get(section_name, {})
        parsed = parse_json_response(section_result.get("response", ""))
        if not isinstance(parsed, dict):
            return []

        events = []
        for event in parsed.get("interesting_events", []):
            events.append(
                {
                    "section": section_name,
                    "frame": event.get("frame"),
                    "event_description": event.get("event_description"),
                    "why_interesting": event.get("why_interesting"),
                    "terrain": event.get("terrain"),
                    "start_seconds": section_info.get("start_seconds"),
                    "end_seconds": section_info.get("end_seconds"),
                    "frame_seconds": self._frame_seconds(event.get("frame")),
                }
            )
        return events

    def _extract(self) -> None:
        """Split frames into sections and detect noteworthy events with the VLM.

        Sections whose result.json already exists are skipped, so an
        interrupted job can be resumed.
        """
        # Split the source frames into model-sized sections.
        config = self.config

        sections = self.frame_parser.create_section_dir()
        print(f"  Created {len(sections)} sections in {config['sections_dir']}")

        # Per-section elapsed seconds, computed from frame timestamps.
        manifest = self._load_section_manifest()

        # Keep intermediate results in the configured output directory.
        output_dir = Path(self.config["output"])
        output_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[extract] Querying VLM (backend={config.get('backend', 'local')}, "
            f"model={config['model']}) ..."
        )

        all_results = []
        for index, curr_section in enumerate(sections, start=1):
            result_path = output_dir / f"{curr_section.name}_output" / "result.json"

            # Reuse completed sections after an interrupted run.
            if result_path.is_file():
                with open(result_path) as result_file:
                    section_result = json.load(result_file)
                if "events" not in section_result:
                    section_result["events"] = self._events_with_seconds(
                        section_result, manifest
                    )
                    with open(result_path, "w") as result_file:
                        json.dump(section_result, result_file, indent=2)
                all_results.append(section_result)
                print(f"  [{index}/{len(sections)}] {curr_section.name} (cached)")
                continue

            # Query the VLM and persist this section immediately.
            response = self.ai_parser.call_vlm(self.prompt, curr_section)
            section_result = {
                "section": curr_section.name,
                "response": response,
                "events": self._events_with_seconds(
                    {"section": curr_section.name, "response": response},
                    manifest,
                ),
            }

            # make and store the result
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, "w") as result_file:
                json.dump(section_result, result_file, indent=2)

            all_results.append(section_result)
            print(f"  [{index}/{len(sections)}] {curr_section.name}")

        # Save an aggregate view of all section responses.
        with open(output_dir / "all_results.json", "w") as result_file:
            json.dump(all_results, result_file, indent=2)

        print(f"  Saved {len(all_results)} section responses to {config['output']}")

    def _load_section_results(self) -> list[dict]:
        """Load persisted section results from the extract stage."""
        # Locate section outputs in chronological filename order.
        output_dir = Path(self.config["output"])
        result_paths = sorted(output_dir.glob("section_*_output/result.json"))
        if not result_paths:
            raise SystemExit(
                f"No section results found in {output_dir}. Run the extract stage first."
            )

        # Read each persisted section response.
        results = []
        for result_path in result_paths:
            with open(result_path) as result_file:
                results.append(json.load(result_file))
        return results

    def _output_dir(self) -> Path:
        """Return the configured output directory, creating it if needed."""
        output_dir = Path(self.config["output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _attach_seconds(self, events: list[dict], manifest: dict) -> None:
        """Re-attach section and frame seconds to merged events.

        The merged events may have lost their seconds, and the merge model
        should never be trusted to produce them, so they are recomputed from
        the manifest and the frame timestamps.
        """
        section_infos = manifest.get("sections") or {}
        for event in events:
            section_info = section_infos.get(event.get("section"), {})
            event["start_seconds"] = section_info.get("start_seconds")
            event["end_seconds"] = section_info.get("end_seconds")
            event["frame_seconds"] = self._frame_seconds(event.get("frame"))

    def _review(self) -> None:
        """Sort detections by frame timestamp and merge them with the LLM."""
        # Load and normalize all extracted section events.
        config = self.config
        section_results = self._load_section_results()
        print(f"[review] Loaded {len(section_results)} section results")

        # Per-section elapsed seconds, computed from frame timestamps.
        manifest = self._load_section_manifest()

        events = []
        unparsed = 0
        for section_result in section_results:
            # Prefer the parsed events (with seconds) stored by the extract
            # stage; fall back to parsing the raw response for older results.
            section_events = section_result.get("events")
            if not isinstance(section_events, list):
                parsed = parse_json_response(section_result.get("response", ""))
                if not isinstance(parsed, dict):
                    unparsed += 1
                    continue
                section_events = self._events_with_seconds(section_result, manifest)
            events.extend(section_events)

        if unparsed:
            print(
                f"  Warning: {unparsed} section responses were not valid JSON "
                "and were skipped"
            )

        if not events:
            raise SystemExit(
                "No interesting events detected across sections; nothing to review."
            )

        # Sort before assigning IDs or asking the LLM to merge events.
        events.sort(key=_frame_sort_key)
        for event_id, event in enumerate(events, start=1):
            event["event_id"] = event_id

        print(f"[review] Merging and filtering {len(events)} detections with LLM ...")

        # Merge and filter bounded windows to keep prompts manageable.
        window_size = int(config.get("review_window", 50))
        reviewed_events = []
        for start in range(0, len(events), window_size):
            window = events[start : start + window_size]
            review_input = (
                f"{self.filter_prompt}\n\n"
                "Detections, already in chronological order:\n"
                f"{json.dumps(window, indent=2)}"
            )
            # Ask the LLM to merge duplicates and remove spurious detections.
            response = self.ai_parser.call_llm(review_input)
            parsed = parse_json_response(response)

            if isinstance(parsed, dict) and isinstance(parsed.get("events"), list):
                reviewed_events.extend(parsed["events"])
            else:
                # Preserve the original window if filtering fails.
                print("  Warning: review response was not valid JSON; keeping events unmerged")
                reviewed_events.extend(window)

        # Renumber events after merging changes their sequence.
        for event_id, event in enumerate(reviewed_events, start=1):
            event["event_id"] = event_id

        # Re-attach per-section and frame seconds in case the review model
        # altered them.
        self._attach_seconds(reviewed_events, manifest)

        # Persist the reviewed event list for review and later stages.
        output_dir = self._output_dir()
        events_out = {
            "frames_dir": config["frames_dir"],
            "sections": len(section_results),
            "events": reviewed_events,
        }
        with open(output_dir / "events.json", "w") as events_file:
            json.dump(events_out, events_file, indent=2)
        print(f"  Saved reviewed events ({len(reviewed_events)}) to {output_dir / 'events.json'}")

    def _generate_questions(self) -> None:
        """Generate sparse-event questions from the human-verified events.json."""
        # Require events produced and reviewed by the earlier stage.
        config = self.config
        output_dir = self._output_dir()
        events_path = output_dir / "events.json"

        if not events_path.is_file():
            raise SystemExit(
                f"No events found at {events_path}. Run the review stage first."
            )

        # Load the verified events.
        with open(events_path) as events_file:
            events_doc = json.load(events_file)

        events = events_doc.get("events", [])
        if not events:
            raise SystemExit("events.json contains no events; nothing to generate questions from.")

        print(f"[questions] Generating questions from {len(events)} reviewed events ...")

        # Send only the fields needed for question generation.
        question_events = [
            {
                "event_id": event.get("event_id"),
                "section": event.get("section"),
                "frame": event.get("frame"),
                "event_description": event.get("event_description"),
                "why_interesting": event.get("why_interesting"),
                "terrain": event.get("terrain"),
                "frame_seconds": event.get("frame_seconds"),
            }
            for event in events
        ]
        question_input = (
            f"{self.question_prompt}\n\n"
            "Verified events, in chronological order:\n"
            f"{json.dumps(question_events, indent=2)}"
        )
        # Ask the LLM to produce sparse-event questions.
        response = self.ai_parser.call_llm(
            question_input,
            max_tokens=config.get("question_max_tokens"),
        )

        # Keep raw output available if JSON parsing fails.
        questions = parse_json_response(response)
        if questions is None:
            print("  Warning: question generation response was not valid JSON")
            questions = {"raw_response": response}

        questions_path = output_dir / "questions.json"
        with open(questions_path, "w") as questions_file:
            json.dump(questions, questions_file, indent=2)
        print(f"  Saved questions to {questions_path}")
