"""Tests for the frozen split loader (gate 2)."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.splits import SplitIntegrityError, load_split, split_sha

BENCH_SHA = "ae3e2429eb08876577f1d871c0108252cc236fa53bf2c6985f1f70f1755e1a8f"


class TestLoadSplit(unittest.TestCase):
    def test_bench_v1_98_rows_and_scenes(self):
        rows = load_split("bench_v1_98")
        self.assertEqual(len(rows), 98)
        scenes = {r["scene"] for r in rows}
        self.assertEqual(len(scenes), 41)

    def test_rows_are_dicts_with_question_column(self):
        rows = load_split("bench_v1_98")
        self.assertIn("msp_question", rows[0])
        self.assertTrue(all(r["msp_question"].strip() for r in rows))

    def test_unknown_split_raises(self):
        with self.assertRaises(SplitIntegrityError):
            load_split("no_such_split")

    def test_tampered_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            splits_dir = tmp / "splits"
            splits_dir.mkdir()
            shutil.copy(REPO_ROOT / "splits" / "bench_v1_98.csv",
                        splits_dir / "bench_v1_98.csv")
            shutil.copy(REPO_ROOT / "splits" / "MANIFEST.json",
                        splits_dir / "MANIFEST.json")
            # Tamper: append one byte to the temp copy.
            with open(splits_dir / "bench_v1_98.csv", "ab") as f:
                f.write(b"x")
            manifest_path = splits_dir / "MANIFEST.json"
            with self.assertRaises(SplitIntegrityError) as ctx:
                load_split("bench_v1_98", manifest_path=manifest_path)
            self.assertIn("INTEGRITY", str(ctx.exception))

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            splits_dir = tmp / "splits"
            splits_dir.mkdir()
            shutil.copy(REPO_ROOT / "splits" / "MANIFEST.json",
                        splits_dir / "MANIFEST.json")
            with self.assertRaises(SplitIntegrityError):
                load_split("bench_v1_98",
                           manifest_path=splits_dir / "MANIFEST.json")


class TestSplitSha(unittest.TestCase):
    def test_split_sha_matches_manifest(self):
        with open(REPO_ROOT / "splits" / "MANIFEST.json") as f:
            manifest = json.load(f)
        pinned = manifest["splits"]["bench_v1_98"]["sha256"]
        self.assertEqual(split_sha("bench_v1_98"), pinned)
        self.assertEqual(split_sha("bench_v1_98"), BENCH_SHA)


if __name__ == "__main__":
    unittest.main()
