# re — strip markdown fences from model output before JSON parsing
# os / os.path — check files exist, join directory paths
# json — parse the model's JSON response
# glob — find frame_*.jpg files inside a directory
# base64 — encode binary JPEG data as ASCII text so it can travel in a JSON HTTP body
# asyncio — run multiple API calls concurrently and bound them with a Semaphore
# anthropic / AsyncAnthropic — talk to the opencode.ai gateway (Anthropic-compatible API)
import re
import os
import json
import glob
import base64
import asyncio

import anthropic
from anthropic import AsyncAnthropic

# The opencode gateway exposes an Anthropic-compatible Messages endpoint at this base URL.
# The SDK internally appends /v1/messages, so we strip any trailing /v1 suffix on our end
# to avoid a double path like /v1/v1/messages.
DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"

# The model name as recognised by the opencode gateway.  Must be a vision-capable
# model (the frames are sent as base64 images in the request).
DEFAULT_MODEL = "qwen3.7-plus"

# When loading frames from a directory, we only pick up files matching this pattern
# (the naming convention used by video_parser.py).  Sorting alphabetically gives
# chronological order because the frame index is zero-padded.
FRAME_GLOB = "frame_*.jpg"


class AIEventParser:
    """Call a VLM via the opencode Anthropic-compatible endpoint and parse the result.

    Usage:
        parser = AIEventParser()
        result = await parser.query(["frame_00001.jpg", "frame_00002.jpg"])
        # result == {"interesting_events": [...], ...}
    """

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 max_concurrent_task: int = 3, api_key: str = None,
                 base_url: str = DEFAULT_BASE_URL, max_retries: int = 2,
                 request_timeout: int = 600):

        # --- API authentication ---
        # The key can be passed directly or read from the OPENCODE_API_KEY env var.
        self.api_key = api_key or os.environ.get("OPENCODE_API_KEY")
        if not self.api_key:
            raise ValueError("OPENCODE_API_KEY not found, provide --api-key or set env var")

        # --- HTTP client ---
        # We use the official Anthropic Python SDK.  The opencode gateway speaks the
        # same wire protocol, so we just point base_url at the gateway.
        # .rstrip("/").removesuffix("/v1") handles base URLs like
        # "https://opencode.ai/zen/go/v1" or "https://opencode.ai/zen/go/v1/" — the
        # SDK will append /v1/messages on its own.
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=base_url.rstrip("/").removesuffix("/v1"),
        )
        self.model_name = model_name

        # --- Concurrency control ---
        # asyncio.Semaphore acts like a bouncer: at most max_concurrent_task API calls
        # can be in flight at the same time.  Each call does `async with self.semaphore`
        # before hitting the network, so it waits its turn if too many are already running.
        # This prevents rate-limit 429 errors and keeps memory bounded.
        self.semaphore = asyncio.Semaphore(max_concurrent_task)

        # If a call fails with a transient error (network hiccup, 429, 5xx), we retry
        # up to this many times with exponential backoff before giving up.
        self.max_retries = max_retries

        # How long to wait for a single API call before timing out.  VLM calls with
        # many images attached can take minutes, so 600 s is a generous default.
        self.request_timeout = request_timeout

        # --- Default system prompt ---
        # This is the instruction we send alongside the frame images.  It tells the
        # model what to look for and what JSON format to return.  You can override it
        # per query by passing a custom `prompt` argument to query().
        self.prompt = """You are analyzing frames from a dashcam video, sampled in chronological order
from one section of a longer drive through forest, rural and tunnel road
environments.

You are looking for rare or noteworthy events: unusual occurrences, unexpected
objects, events, sudden changes, unique terrain transitions, or anything that
stands out from just regularly driving. DO NOT report normal driving, regular
traffic, or mundane scenery. We only want unusual, rare and interesting events.

Each frame you receive is a file in its own section folder, in the order
frame_00000.jpg, frame_00001.jpg, ... The frame index is the part of the
filename before ".jpg". Report events by referring to the specific frame file
they occur in, along with the section_id you were told for this batch. We keep
these files on disk, so the exact frame name lets us verify the event later.

Respond with ONLY a JSON object (no markdown, no commentary). If nothing
noteworthy happens, return an empty interesting_events list.

{
    "interesting_events": [
        {
        "section_id": <integer section_id you received for this batch>,
        "frame": "name of the frame file, e.g. frame_00042.jpg",
        "event_description": "detailed description of what happened",
        "why_interesting": "why this is considered a rare/sparse event",
        "terrain": "one of: forest, rural, tunnel, other"
        }
    ]
}
"""

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(body: str) -> dict:
        """Try to extract a valid JSON dict from the model's raw text output.

        LLMs are messy — even with careful prompting they sometimes wrap JSON in
        ```json ... ``` markdown fences, or prepend a spurious opening brace, or
        append commentary after the closing brace.  This method tries several
        recovery strategies in order, from most to least specific:

          1. Raw text as-is (the happy path).
          2. Prepend "{" — the model may have dropped the opening brace because
             we pre-filled the assistant message with "{", making the model think
             the brace was already emitted.
          3. Strip markdown fences (``` or ```json) from (1), then parse.
          4. Strip markdown fences from (2), then parse.
          5. Regex-extract the first { ... } block from the stripped text and
             parse that.  This catches cases where the model added text before
             or after the JSON object.

        If every path fails, the last JSONDecodeError is raised and caught by
        query()'s retry loop — the model may produce valid output on the next
        attempt.
        """
        last_error = None
        # Try two "candidates": the raw text, and the raw text with "{" prepended.
        for candidate in (body, "{" + body):
            # Strip ```json fences — just remove the backtick markers, keep everything else.
            cleaned = re.sub(r"```(?:json)?", "", candidate).strip()

            # Attempt 1: direct JSON parse of the cleaned text.
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                last_error = e

            # Attempt 2: find the first { ... } block with a regex and parse just that.
            # re.DOTALL makes "." match newlines, so the regex works across multi-line text.
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    last_error = e

        # Nothing worked — let the caller (the retry loop) handle it.
        raise last_error

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    async def query(self, frame_paths: list, prompt: str = None) -> dict:
        """Send a list of frame images + a prompt to the VLM and return parsed JSON.

        This is the single place where HTTP calls happen.  Everything else
        (query_directory, query_multiple) delegates to this method.

        The flow:
          1. Build a content list: text prompt first, then each frame as a
             base64-encoded image block.
          2. Wrap it in Anthropic Messages format:
             - The user message contains the prompt + images.
             - We pre-fill the assistant's response with "{" to force the model
               to continue in JSON.
          3. Call the API with retries and exponential backoff.
          4. Parse the body with _parse_json and return the dict.

        Args:
            frame_paths: list of paths to JPEG frame files (sent in order).
            prompt: custom prompt.  Uses self.prompt when None.
        """

        # --- Step 1: build the content list ---
        # The Anthropic Messages API expects content as a list of blocks.
        # A block is either {"type": "text", "text": "..."} or
        # {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}.
        content = [{"type": "text", "text": prompt or self.prompt}]

        for fp in frame_paths:
            # Read the JPEG file as raw bytes …
            with open(fp, "rb") as f:
                raw_bytes = f.read()
            # … and encode those bytes as a base64 string.  base64 turns binary
            # data into plain ASCII so it can live inside a JSON request body.
            b64 = base64.standard_b64encode(raw_bytes).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })

        # --- Step 2: build the Messages-format payload ---
        # The "user" message carries the prompt + frames.
        # The "assistant" message is a prefill: by putting "{" as the assistant's
        # first response, the model is forced to continue it — which means it must
        # write valid JSON (starting with the key/value pairs after "{").
        # This avoids most formatting problems entirely.
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "{"}]},
        ]

        # --- Step 3: call the API with retries ---
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Wait for the semaphore — if max_concurrent_task calls are already
                # running, this will pause here until one finishes and releases the slot.
                async with self.semaphore:
                    response = await self.client.messages.create(
                        model=self.model_name,
                        temperature=0.3,  # low temp = more deterministic, fewer hallucinations
                        messages=messages,
                        timeout=self.request_timeout,
                    )

                # The response.content is a list of ContentBlock objects.  We only
                # care about text blocks (not tool-use blocks, etc.), so we join
                # all text block contents into a single string.
                body = "".join(b.text for b in response.content if b.type == "text")

                # Parse the text into a Python dict.
                result = self._parse_json(body)

                # Ensure the result always has "interesting_events" so callers
                # don't have to guard against KeyError.
                result.setdefault("interesting_events", [])
                return result

            # --- Error classification ---
            # Not all errors should be retried the same way.

            except anthropic.APIStatusError as e:
                last_error = e
                # 4xx errors (bad request, bad API key, unknown model) are
                # PERMANENT — retrying with the same payload will just fail again.
                # The one exception is 429 (rate limit), which IS transient.
                if e.status_code < 500 and e.status_code != 429:
                    break  # stop retrying immediately

            except (anthropic.APIConnectionError,   # DNS / network failure
                    anthropic.APITimeoutError,      # request took too long
                    asyncio.TimeoutError,           # client-side timeout
                    json.JSONDecodeError) as e:     # model output wasn't valid JSON
                last_error = e
                # These are all TRANSIENT — the next attempt might succeed.

            # Exponential backoff before the next retry:
            # attempt 0 → sleep 2s, attempt 1 → 4s, attempt 2 → 8s, etc.
            if attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt * 2)

        # All retries exhausted — return a partial result with an error field
        # so the caller knows something went wrong, but the pipeline can still
        # aggregate results from other (successful) queries.
        print(f"Query failed after {attempt + 1} attempt(s): {last_error}")
        return {"interesting_events": [],
                "error": f"{type(last_error).__name__}: {last_error}"}

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def query_directory(self, frames_dir: str, prompt: str = None) -> dict:
        """Scan a directory for frame_*.jpg files and query the model with all of them.

        Sort order is alphabetical, which matches chronological order because
        video_parser.py names frames with zero-padded indices (frame_00000.jpg,
        frame_00001.jpg, ...).
        """
        # glob finds all files matching the pattern; sorted ensures they're in order.
        paths = sorted(glob.glob(os.path.join(frames_dir, FRAME_GLOB)))
        if not paths:
            return {"interesting_events": [], "error": f"no frames in {frames_dir}"}
        return await self.query(paths, prompt)

    async def query_multiple(self, batch_specs: list) -> list:
        """Fire multiple queries at once, all sharing the same semaphore.

        Use this when you have several independent sets of frames (e.g., different
        sections of a video) and want to process them concurrently rather than
        one-after-the-other.

        Each item in batch_specs is a dict with:
            frame_paths  — list of file paths (required)
            prompt       — optional per-batch prompt override

        Returns a list of result dicts, one per batch, in the same order as
        batch_specs.  The semaphore ensures we never exceed max_concurrent_task
        in-flight calls, even if batch_specs has 100 entries.
        """
        return await asyncio.gather(
            *[self.query(b["frame_paths"], b.get("prompt")) for b in batch_specs]
        )
