"""Convert event frame ranges in timeline.json to elapsed video seconds."""

import argparse
import json
from pathlib import Path

from pipeline.timeframe_converter import TimeframeConverter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add start_seconds and end_seconds to a timeline JSON file."
    )
    parser.add_argument("--timeline", default="timeline.json")
    parser.add_argument("--output", default="timeline_with_seconds.json")
    args = parser.parse_args()

    timeline_path = Path(args.timeline)
    with timeline_path.open() as timeline_file:
        timeline = json.load(timeline_file)

    try:
        converter = TimeframeConverter(timeline["frames_dir"])
    except (KeyError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Could not initialize timeframe converter: {error}") from error

    converted_events = []
    for event in timeline.get("events", []):
        try:
            seconds = converter.timeframe_to_seconds(
                event["start_frame"],
                event["end_frame"],
            )
        except (KeyError, ValueError) as error:
            event_id = event.get("event_id", "unknown")
            raise SystemExit(f"Could not convert event {event_id}: {error}") from error

        converted_event = dict(event)
        converted_event.update(seconds)
        converted_events.append(converted_event)

    converted_timeline = dict(timeline)
    converted_timeline["events"] = converted_events
    output_path = Path(args.output)
    with output_path.open("w") as output_file:
        json.dump(converted_timeline, output_file, indent=2)

    print(f"Converted {len(converted_events)} events to {output_path}")


if __name__ == "__main__":
    main()
