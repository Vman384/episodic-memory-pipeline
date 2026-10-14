import json
import tempfile
import unittest
from pathlib import Path

from generate_duration_stats import (
    SequenceStats,
    boreas_spec,
    collect_sequence_stats,
    duration_group,
    render_latex_table,
    summarize_sequences,
)


class DurationStatsTests(unittest.TestCase):
    def test_duration_bins_use_inclusive_upper_boundaries(self):
        self.assertEqual(duration_group(10 * 60), "Short")
        self.assertEqual(duration_group(10 * 60 + 1), "Medium")
        self.assertEqual(duration_group(15 * 60), "Medium")
        self.assertEqual(duration_group(15 * 60 + 1), "Long")

    def test_reads_timestamp_duration_and_question_counts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)
            sequence_name = "boreas-test-sequence"
            spec = boreas_spec(data_root, sequence_name)
            spec.frames_dir.mkdir(parents=True)

            start_timestamp = 1_700_000_000_000_000
            (spec.frames_dir / f"{start_timestamp}.png").touch()
            (spec.frames_dir / f"{start_timestamp + 600_000_000}.png").touch()

            spec.sparse_questions.parent.mkdir(parents=True)
            spec.sparse_questions.write_text(
                json.dumps(
                    {
                        "false_events": [{"description": "not counted as a QA"}],
                        "questions": [{"type": "deceptive"}, {"type": "existence"}],
                    }
                ),
                encoding="utf-8",
            )
            spec.temporal_questions.parent.mkdir(parents=True)
            spec.temporal_questions.write_text(
                json.dumps({"questions": [{"type": "pairwise_order"}]}),
                encoding="utf-8",
            )

            stats = collect_sequence_stats(spec)

        self.assertEqual(stats.duration_seconds, 600)
        self.assertEqual(stats.sparse_question_count, 2)
        self.assertEqual(stats.temporal_question_count, 1)
        self.assertEqual(stats.total_question_count, 3)

    def test_latex_table_aggregates_questions_and_prints_empty_bins(self):
        sequences = [
            SequenceStats(
                name="short-drive",
                duration_seconds=600,
                sparse_question_count=2,
                temporal_question_count=3,
            )
        ]
        rows = summarize_sequences(sequences)
        table = render_latex_table(sequences)

        self.assertEqual(rows[0].video_count, 1)
        self.assertEqual(rows[0].total_question_count, 5)
        self.assertEqual(rows[-1].video_count, 1)
        self.assertIn(r"Short & 1 & 10.00 & 2 & 3 & 5", table)
        self.assertIn(r"Medium & 0 & -- & 0 & 0 & 0", table)
        self.assertIn(r"Long & 0 & -- & 0 & 0 & 0", table)
        self.assertIn(r"\begin{table}[!t]", table)
        self.assertIn(r"\resizebox{\columnwidth}{!}{%", table)
        self.assertIn(r"\label{tab:duration_composition}", table)


if __name__ == "__main__":
    unittest.main()
