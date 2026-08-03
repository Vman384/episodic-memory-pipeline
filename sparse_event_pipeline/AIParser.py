"""Send frame subsections to a VLM through the OpenCode Go API."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from anthropic import APIStatusError, AsyncAnthropic


DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"

DEFAULT_MODEL = "qwen3.7-plus"

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
        prompt: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_concurrent: int = 3,
        max_tokens: int = 2048,
    ):
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than zero")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        load_dotenv()
        api_key = os.environ.get("OPENCODE_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENCODE_API_KEY is not set. Add it to a .env file or "
                "export it in your shell."
            )

        # The SDK appends /v1/messages to its base URL. The public config
        # includes /v1, so remove it before constructing the client.
        sdk_base_url = base_url.rstrip("/").removesuffix("/v1")
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=sdk_base_url,
            timeout=30.0,
            max_retries=0,
        )
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.semaphore = asyncio.Semaphore(max_concurrent)

    @staticmethod
    def _frame_paths(section_dir: Path) -> list[Path]:
        paths = []
        for path in section_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_MEDIA_TYPES:
                continue
            paths.append(path)
        paths.sort(key=lambda path: int(path.stem))
        return paths

    @staticmethod
    def _parse_json(body: str) -> dict:
        last_error = None
        for candidate in (body, "{" + body):
            cleaned = re.sub(r"```(?:json)?", "", candidate).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                last_error = exc
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as exc:
                    last_error = exc
        raise last_error

    async def _make_request(self, messages: list[dict], model: str,
                            max_tokens: int, temperature: float) -> Any:
        """Send a request through the Anthropic Messages SDK."""
        return await self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )

    async def call_llm(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        async with self.semaphore:
            response = await self._make_request(
                messages=[{"role": "user", "content": prompt}],
                model=model or self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=0.2,
            )
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts).strip()

    async def call_vlm(
        self,
        frame_paths: list[Path],
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        label: str = "",
    ) -> dict:
        content = [{"type": "text", "text": prompt}]
        for frame_path in frame_paths:
            raw_bytes = frame_path.read_bytes()
            b64 = base64.b64encode(raw_bytes).decode("ascii")
            media_type = IMAGE_MEDIA_TYPES[frame_path.suffix.lower()]
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type,
                           "data": b64},
            })

        async with self.semaphore:
            if label:
                print(f"    Calling API for {label} ...")

            for attempt in range(3):
                try:
                    data = await self._make_request(
                        messages=[{"role": "user", "content": content}],
                        model=model or self.model,
                        max_tokens=max_tokens or self.max_tokens,
                        temperature=0.3,
                    )
                    break
                except APIStatusError as exc:
                    delay = 3
                    print(f"    API error on attempt {attempt + 1}/3: "
                          f"{exc.status_code}. "
                          f"Retrying in {delay}s ...")
                    await asyncio.sleep(delay)
            else:
                raise RuntimeError("All 3 API attempts failed")

        parts = []
        for block in data.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        body = "".join(parts).strip()

        preview = body[:500].replace("\n", "\\n")
        if len(body) > 500:
            preview += f" ... ({len(body)} chars total)"
        print(f"    Response ({len(body)} chars): {preview}")

        try:
            return self._parse_json(body)
        except json.JSONDecodeError:
            print(f"    Raw response that failed to parse:\n{body}")
            raise

    async def _query_section(self, section_dir: Path) -> dict:
        frame_paths = self._frame_paths(section_dir)
        prompt = f"Section name: {section_dir.name}\n\n{self.prompt}"
        return await self.call_vlm(
            frame_paths, prompt,
            label=f"{section_dir.name} ({len(frame_paths)} frames)",
        )

    async def process_subsections(
        self,
        sections_dir: str | Path,
        output_dir: str | Path,
    ) -> list[Path]:
        sections_dir = Path(sections_dir).expanduser()
        output_dir = Path(output_dir).expanduser()

        if not sections_dir.is_dir():
            raise FileNotFoundError(f"Sections directory not found: {sections_dir}")

        section_dirs = []
        for path in sections_dir.iterdir():
            if path.is_dir():
                section_dirs.append(path)
        section_dirs.sort()

        output_dir.mkdir(parents=True, exist_ok=True)

        aggregated = []
        total = len(section_dirs)
        print(f"Processing {total} sections with up to {self.semaphore._value} "
              f"concurrent calls ...")

        async def process(section_dir: Path, index: int,
                          total: int) -> tuple[Path, dict]:
            result_dir = output_dir / f"{section_dir.name}_output"
            result_dir.mkdir(parents=True, exist_ok=True)

            frames = self._frame_paths(section_dir)
            print(f"[{index}/{total}] Starting {section_dir.name} "
                  f"({len(frames)} frames)")

            result = await self._query_section(section_dir)
            result_path = result_dir / "result.json"
            result_path.write_text(json.dumps(result, indent=2) + "\n")

            entry = {
                "section": section_dir.name,
                "num_frames": len(frames),
                "start_frame": frames[0].name if frames else None,
                "end_frame": frames[-1].name if frames else None,
            }
            entry.update(result)
            aggregated.append(entry)

            print(f"[{index}/{total}] {section_dir.name} done")
            return result_path, entry

        results = await asyncio.gather(
            *(process(section, i + 1, total)
              for i, section in enumerate(section_dirs))
        )

        result_paths = [path for path, _ in results]

        aggregate_path = output_dir / "all_results.json"
        aggregate_path.write_text(json.dumps(aggregated, indent=2) + "\n")
        print(f"Wrote aggregate results to {aggregate_path}")

        return result_paths
