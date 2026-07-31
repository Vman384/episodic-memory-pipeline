import os
import json
import asyncio
import argparse

from AIParser import AIEventParser, DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser(
        description="Query a VLM with frame images via the opencode gateway."
    )
    ap.add_argument("input", help="Directory of frame_*.jpg files OR a single frame file")
    ap.add_argument("--prompt", default=None, help="Custom prompt (default: built-in event prompt)")
    ap.add_argument("--output", default=None, help="Write JSON response to file")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--api-key", default=None, help="API key (default: OPENCODE_API_KEY env var)")
    ap.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent API calls")
    args = ap.parse_args()

    parser = AIEventParser(
        model_name=args.model,
        max_concurrent_task=args.max_concurrent,
        api_key=args.api_key,
    )

    async def run():
        if os.path.isdir(args.input):
            result = await parser.query_directory(args.input, args.prompt)
        else:
            result = await parser.query([args.input], args.prompt)

        print(json.dumps(result, indent=2))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

    asyncio.run(run())


if __name__ == "__main__":
    main()
