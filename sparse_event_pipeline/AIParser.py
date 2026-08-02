"""Send frame subsections to a VLM through the OpenCode Go API."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic


DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "qwen3.7-plus"
PROMPT = """You are analyzing frames from a dashcam video, sampled in chronological order
from one section of a longer drive through forest, rural and tunnel road
environments.

Look for rare or noteworthy events: unusual occurrences, unexpected objects,
sudden changes, unique terrain transitions, or anything that stands out from
normal driving. Do not report normal driving, regular traffic, or mundane
scenery.

Respond with ONLY a JSON object. If nothing noteworthy happens, return an
empty interesting_events list.

{
    "interesting_events": [
        {
            "section_id": "the section name",
            "frame": "the exact frame filename",
            "event_description": "what happened",
            "why_interesting": "why this is unusual",
            "terrain": "forest, rural, tunnel, or other"
        }
    ]
}
"""

# Maps lowercase file extensions to the corresponding MIME type string
# expected by the Anthropic-compatible Messages API.
IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class AIEventParser:
    """Process each frame subsection concurrently with a VLM."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        prompt: str = PROMPT,
        max_concurrent: int = 3,
        max_tokens: int = 2048,
    ):
        # Sanity checks so nonsense values fail fast.
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than zero")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        # The API key can come from the environment or the constructor.
        self.api_key = api_key or os.environ.get("OPENCODE_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENCODE_API_KEY or pass api_key explicitly")

        # The OpenCode Go gateway speaks the Anthropic wire protocol, so we
        # use the official SDK pointed at the gateway base URL.
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=base_url.rstrip("/").removesuffix("/v1"),
        )
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens

        # Bounded concurrency: at most max_concurrent API calls can be in
        # flight at once, avoiding rate limits and memory pressure.
        self.semaphore = asyncio.Semaphore(max_concurrent)

    @staticmethod
    def _frame_paths(section_dir: Path) -> list[Path]:
        """Return supported numeric frame files in chronological order.

        Only regular files whose extension is in IMAGE_MEDIA_TYPES are kept.
        Frames are sorted by their numeric stem so the VLM sees them in the
        order they were captured.
        """
        # Collect every file that looks like a supported image.
        paths = []
        for path in section_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_MEDIA_TYPES:
                continue
            paths.append(path)

        # Sort chronologically by the numeric filename stem.
        paths.sort(key=lambda path: int(path.stem))
        return paths

    async def call_llm(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a text-only prompt and return the raw text response.

        No images are attached — use this for story generation, timeline
        summarisation, or any text-in/text-out task.  The caller is
        responsible for parsing JSON from the response if needed.

        Args:
            prompt: The text prompt to send.
            model: Override the instance model (e.g. use a cheaper text
                   model instead of the VLM).  Defaults to ``self.model``.
            max_tokens: Override max output tokens.
        """
        async with self.semaphore:
            response = await self.client.messages.create(
                model=model or self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )

        body_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                body_parts.append(block.text)
        return "".join(body_parts).strip()

    async def call_vlm(
        self,
        frame_paths: list[Path],
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Send a list of frame images with a prompt and return parsed JSON.

        Each frame is read from disk and base64-encoded.  The prompt is
        sent as a leading text block, followed by one image block per
        frame in the order given.  The response is parsed as JSON and
        returned as a dict.

        Args:
            frame_paths: Frame image files to attach (sent in order).
            prompt: Text prompt to accompany the images.
            model: Override the instance model.
            max_tokens: Override max output tokens.
        """
        content = [{"type": "text", "text": prompt}]

        for frame_path in frame_paths:
            raw_bytes = frame_path.read_bytes()
            image_data = base64.b64encode(raw_bytes).decode("ascii")
            media_type = IMAGE_MEDIA_TYPES[frame_path.suffix.lower()]

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                }
            )

        async with self.semaphore:
            response = await self.client.messages.create(
                model=model or self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=0.3,
                messages=[{"role": "user", "content": content}],
            )

        body_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                body_parts.append(block.text)
        body = "".join(body_parts).strip()

        return json.loads(body)

    async def _query_section(self, section_dir: Path) -> dict:
        """Send one subsection to the VLM and return the parsed JSON result.

        Convenience wrapper around :meth:`call_vlm` that discovers and
        sorts the frame files inside *section_dir* and prepends the
        section name to the instance prompt.
        """
        frame_paths = self._frame_paths(section_dir)
        prompt = f"Section name: {section_dir.name}\n\n{self.prompt}"
        return await self.call_vlm(frame_paths, prompt)

    async def process_subsections(
        self,
        sections_dir: str | Path,
        output_dir: str | Path,
    ) -> list[Path]:
        """Process every direct subsection and save one JSON result per section.

        Each subsection directory under *sections_dir* is queried
        concurrently.  Results are written to
        ``<output_dir>/<section_name>_output/result.json``.
        """
        sections_dir = Path(sections_dir).expanduser()
        output_dir = Path(output_dir).expanduser()

        if not sections_dir.is_dir():
            raise FileNotFoundError(f"Sections directory not found: {sections_dir}")

        # Discover every subdirectory under the sections root.
        section_dirs = []
        for path in sections_dir.iterdir():
            if path.is_dir():
                section_dirs.append(path)
        section_dirs.sort()

        # Ensure the output root exists before writing results.
        output_dir.mkdir(parents=True, exist_ok=True)

        async def process(section_dir: Path) -> Path:
            # One isolated output folder per section.
            result_dir = output_dir / f"{section_dir.name}_output"
            result_dir.mkdir(parents=True, exist_ok=True)

            # Call the VLM for this section's frames.
            result = await self._query_section(section_dir)

            # Persist the JSON result inside the section output folder.
            result_path = result_dir / "result.json"
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            return result_path

        # Launch all section queries concurrently; the semaphore limits
        # how many actually call the API at the same time.
        return await asyncio.gather(
            *(process(section) for section in section_dirs)
        )


def main() -> None:
    """Run the subsection VLM parser from the command line."""
    parser = argparse.ArgumentParser(
        description="Process frame subsections with an OpenCode Go VLM."
    )
    parser.add_argument("sections_dir", help="Directory containing subsection folders")
    parser.add_argument("--output", required=True, help="Directory for subsection results")
    parser.add_argument("--prompt", default=PROMPT, help="Prompt sent to the VLM")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="VLM model name")
    parser.add_argument("--api-key", default=None, help="API key or OPENCODE_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenCode Go API URL")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent VLM requests (default: 3)",
    )
    args = parser.parse_args()

    # Wire up the parser from CLI arguments.
    parser_instance = AIEventParser(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        prompt=args.prompt,
        max_concurrent=args.max_concurrent,
    )

    # Run the async pipeline and print every produced result path.
    result_paths = asyncio.run(
        parser_instance.process_subsections(args.sections_dir, args.output)
    )
    for result_path in result_paths:
        print(result_path)


if __name__ == "__main__":
    main()
