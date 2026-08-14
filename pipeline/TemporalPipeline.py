"""Temporal order pipeline: extract, build timeline, generate questions."""

import json
from pathlib import Path

from pipeline.AIParser import AIParser
from pipeline.ConfigLoader import ConfigLoader
from pipeline.frame_parser import FrameParser


def parse_json_response(response: str):
    """Best-effort parse of JSON embedded in a model response.

    Strips markdown code fences and returns the outermost JSON object, or
    None when no valid JSON object is found.
    """
    # Remove optional markdown wrapping.
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Extract the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None

    # Treat malformed model output as unavailable data.
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _frame_sort_key(event: dict):
    """Sort key ordering events by start frame timestamp.

    Frame filenames are epoch timestamps, so the numeric stem gives global
    chronological order. Events without a parseable frame go last.
    """
    # Numeric frame names preserve order; invalid frames sort last.
    try:
        return (0, int(Path(str(event.get("start_frame", ""))).stem))
    except ValueError:
        return (1, 0)


class TemporalPipeline:
    """
    Pipeline to assess temporal order memory.

    Stages:
        extract   - split frames into sections and summarise each with the VLM
        timeline  - sort section events by frame timestamp, merge them with
                    the LLM, and write timeline.json plus a prose storyline
        questions - generate temporal questions from the human-verified
                    timeline.json
    """

    STAGES = ("extract", "timeline", "questions")

    def __init__(
        self,
        config_path: str | Path,
        prompt: str,
        filter_prompt: str,
        storyline_prompt: str,
        question_prompt: str,
    ):
        # Store prompts; helpers are created when config loads.
        self.config_path = config_path
        self.prompt = prompt
        self.filter_prompt = filter_prompt
        self.storyline_prompt = storyline_prompt
        self.question_prompt = question_prompt
        self.config = None
        self.frame_parser = None
        self.ai_parser = None

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
        )

        # Initialize the configured local or API backend.
        self.ai_parser = AIParser(config=self.config)
        return self.config

    def run(self, stage: str | None = None) -> None:
        """
        Run one temporal pipeline stage. When stage is None, run extract
        followed by timeline. The questions stage must be run separately,
        after a human has reviewed timeline.json.
        """
        # Validate the requested stage before loading models.
        if stage is not None and stage not in self.STAGES:
            raise ValueError(f"stage must be one of {self.STAGES}")

        # Prepare frame and model helpers for the selected stages.
        self.load_config()

        if stage is None or stage == "extract":
            # call VLM to extract section of files.
            self._extract()
        if stage is None or stage == "timeline":
            # call LLM to build a timeline
            self._build_timeline()
        if stage == "questions":
            self._generate_questions()


    def _extract(self) -> None:
        """Split frames into sections and summarise each one with the VLM.

        Sections whose result.json already exists are skipped, so an
        interrupted job can be resumed.
        """
        # Split the source frames into model-sized sections.
        config = self.config

        sections = self.frame_parser.create_section_dir()
        print(f"  Created {len(sections)} sections in {config['sections_dir']}")

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
                    all_results.append(json.load(result_file))
                print(f"  [{index}/{len(sections)}] {curr_section.name} (cached)")
                continue

            # Query the VLM and persist this section immediately.
            response = self.ai_parser.call_vlm(self.prompt, curr_section)
            section_result = {
                "section": curr_section.name,
                "response": response,
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

    def _build_timeline(self) -> None:
        """Sort section events by frame timestamp and merge them with the LLM."""
        # Load and normalize all extracted section events.
        config = self.config
        section_results = self._load_section_results()
        print(f"[timeline] Loaded {len(section_results)} section results")

        events = []
        unparsed = 0
        for section_result in section_results:
            parsed = parse_json_response(section_result["response"])
            if not isinstance(parsed, dict):
                unparsed += 1
                continue

            # Retain only fields needed by later timeline stages.
            for event in parsed.get("events", []):
                events.append(
                    {
                        "section": section_result["section"],
                        "start_frame": event.get("start_frame"),
                        "end_frame": event.get("end_frame"),
                        "description": event.get("description"),
                    }
                )

        if unparsed:
            print(
                f"  Warning: {unparsed} section responses were not valid JSON "
                "and were skipped"
            )

        # Sort before assigning IDs or asking the LLM to merge events.
        events.sort(key=_frame_sort_key)
        for event_id, event in enumerate(events, start=1):
            event["event_id"] = event_id

        print(f"[timeline] Merging {len(events)} events with LLM ...")

        # Merge bounded windows to keep prompts manageable.
        window_size = int(config.get("merge_window", 50))
        merged_events = []
        for start in range(0, len(events), window_size):
            window = events[start : start + window_size]
            merge_input = (
                f"{self.filter_prompt}\n\n"
                "Events, already in chronological order:\n"
                f"{json.dumps(window, indent=2)}"
            )
            # Ask the LLM to remove duplicates without reordering events.
            response = self.ai_parser.call_llm(merge_input)
            parsed = parse_json_response(response)

            if isinstance(parsed, dict) and isinstance(parsed.get("events"), list):
                merged_events.extend(parsed["events"])
            else:
                # Preserve the original window if merging fails.
                print("  Warning: merge response was not valid JSON; keeping events unmerged")
                merged_events.extend(window)

        # Renumber events after merging changes their sequence.
        for event_id, event in enumerate(merged_events, start=1):
            event["event_id"] = event_id

        # Persist the merged timeline for review and later stages.
        output_dir = self._output_dir()
        timeline = {
            "frames_dir": config["frames_dir"],
            "sections": len(section_results),
            "events": merged_events,
        }
        with open(output_dir / "timeline.json", "w") as timeline_file:
            json.dump(timeline, timeline_file, indent=2)
        print(f"  Saved timeline ({len(merged_events)} events) to {output_dir / 'timeline.json'}")

        # Do not call the storyline model when no events remain.
        if not merged_events:
            print("  No events to build a storyline from")
            return

        print("[timeline] Building storyline with LLM ...")
        # Convert the reviewed chronological events into prose.
        storyline_input = (
            f"{self.storyline_prompt}\n\n"
            "Chronological timeline events:\n"
            f"{json.dumps(merged_events, indent=2)}"
        )
        storyline = self.ai_parser.call_llm(
            storyline_input,
            max_tokens=config.get("storyline_max_tokens"),
        )
        with open(output_dir / "storyline.txt", "w") as storyline_file:
            storyline_file.write(storyline)
        print(f"  Saved storyline to {output_dir / 'storyline.txt'}")

    def _generate_questions(self) -> None:
        """Generate temporal questions from the human-verified timeline.json."""
        # Require a timeline produced and reviewed by the earlier stage.
        config = self.config
        output_dir = self._output_dir()
        timeline_path = output_dir / "timeline.json"

        if not timeline_path.is_file():
            raise SystemExit(
                f"No timeline found at {timeline_path}. Run the timeline stage first."
            )

        # Load the verified timeline events.
        with open(timeline_path) as timeline_file:
            timeline = json.load(timeline_file)

        events = timeline.get("events", [])
        if not events:
            raise SystemExit("Timeline contains no events; nothing to generate questions from.")

        print(f"[questions] Generating questions from {len(events)} timeline events ...")

        # Send only the fields needed for question generation.
        question_events = [
            {
                "event_id": event.get("event_id"),
                "start_frame": event.get("start_frame"),
                "end_frame": event.get("end_frame"),
                "description": event.get("description"),
            }
            for event in events
        ]
        question_input = (
            f"{self.question_prompt}\n\n"
            "Verified timeline events, in chronological order:\n"
            f"{json.dumps(question_events, indent=2)}"
        )
        # Ask the LLM to produce temporal-order questions.
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
