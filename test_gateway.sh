#!/bin/bash

# Checks that the OpenCode gateway keeps a conversation across requests
# (previous_response_id). Run from the repository root:
#   OPENCODE_API_KEY=... bash test_gateway.sh
# Set FRAME_DIR to a camera folder to also send its first frame and report the
# tokens one image adds to the conversation.
module load python3/3.11.7

# Enter the environment containing the local vLLM pipeline dependencies.
source /scratch/pg06/vm4618/envs/vllm_env/bin/activate
set -euo pipefail

: "${OPENCODE_API_KEY:?Set OPENCODE_API_KEY before running this test}"

PYTHON="${PYTHON:-python3}"

"$PYTHON" - <<'EOF'
import base64
import os
import uuid

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENCODE_API_KEY"],
    base_url="https://opencode.ai/zen/go/v1",
    max_retries=0,
)
reasoning = {"effort": "low"}
headers = {
    "User-Agent": "episodic-memory-pipeline/1.0",
    "x-opencode-session": f"ses_{uuid.uuid4().hex}",
}

first = client.responses.create(
    model="gpt-5.6-luna",
    input="Remember this secret word: PINEAPPLE. Reply with OK only.",
    reasoning=reasoning,
    max_output_tokens=500,
    store=True,
    extra_headers=headers,
)
print("First reply:", first.output_text)

second = client.responses.create(
    model="gpt-5.6-luna",
    previous_response_id=first.id,
    input="What was the secret word?",
    reasoning=reasoning,
    max_output_tokens=500,
    store=True,
    extra_headers=headers,
)
print("Second reply:", second.output_text)
print("Input tokens:", second.usage.input_tokens)

if "PINEAPPLE" in second.output_text.upper():
    print("PASS: the gateway kept the conversation across requests.")
else:
    print("FAIL: the second reply did not recall the secret word.")

frame_dir = os.environ.get("FRAME_DIR")
if frame_dir:
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    frames = sorted(
        name
        for name in os.listdir(frame_dir)
        if os.path.splitext(name)[1].lower() in media_types
    )
    frame_path = os.path.join(frame_dir, frames[0])
    media_type = media_types[os.path.splitext(frame_path)[1].lower()]
    with open(frame_path, "rb") as frame_file:
        frame_bytes = frame_file.read()
    encoded = base64.b64encode(frame_bytes).decode("utf-8")
    print(f"Frame: {frames[0]} ({len(frame_bytes)} bytes)")

    image_reply = client.responses.create(
        model="gpt-5.6-luna",
        previous_response_id=second.id,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "This is one frame from the drive. Reply with OK only."},
                    {"type": "input_image", "image_url": f"data:{media_type};base64,{encoded}"},
                ],
            }
        ],
        reasoning=reasoning,
        max_output_tokens=500,
        store=True,
        extra_headers=headers,
    )
    print("Image reply:", image_reply.output_text)
    print("Input tokens with image:", image_reply.usage.input_tokens)
    print("Tokens added by the image:", image_reply.usage.input_tokens - second.usage.input_tokens)
EOF