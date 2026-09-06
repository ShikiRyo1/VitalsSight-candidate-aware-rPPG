"""Dataset-free regression tests for exact checksums and historical EOL lineage."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "reproducibility/v32_submission/scripts/verify_package.py"
SPEC = importlib.util.spec_from_file_location("v32_public_package_verifier", VERIFIER)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class PublicChecksumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.relative = "frozen_aggregates/fixture.csv"
        self.file = self.root / self.relative
        self.file.parent.mkdir(parents=True)
        self.data = b"method,mae\nsynthetic,1.25\n"
        self.file.write_bytes(self.data)

    def manifest(self, name: str, data: bytes) -> Path:
        path = self.root / name
        digest = hashlib.sha256(data).hexdigest().upper()
        path.write_bytes(f"{digest}  {self.relative}\n".encode("utf-8"))
        return path

    def test_exact_lf_checkout_passes(self) -> None:
        manifest = self.manifest("public.txt", self.data)
        self.assertEqual(verifier.verify_hashes(self.root, manifest), 1)

    def test_public_checksum_rejects_changed_line_endings(self) -> None:
        manifest = self.manifest("public.txt", self.data)
        self.file.write_bytes(self.data.replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            verifier.verify_hashes(self.root, manifest)

    def test_public_checksum_rejects_changed_value(self) -> None:
        manifest = self.manifest("public.txt", self.data)
        self.file.write_bytes(self.data.replace(b"1.25", b"1.26"))
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            verifier.verify_hashes(self.root, manifest)

    def test_public_checksum_rejects_missing_file(self) -> None:
        manifest = self.manifest("public.txt", self.data)
        self.file.unlink()
        with self.assertRaisesRegex(RuntimeError, "file is missing"):
            verifier.verify_hashes(self.root, manifest)

    def test_original_crlf_lineage_is_explicit(self) -> None:
        manifest = self.manifest("historical.txt", self.data.replace(b"\n", b"\r\n"))
        self.assertEqual(
            verifier.verify_historical_aggregate_lineage(self.root, manifest),
            {"exact_original_bytes": 0, "original_crlf_reconstructed": 1},
        )

    def test_lineage_rejects_genuine_content_change(self) -> None:
        manifest = self.manifest("historical.txt", self.data.replace(b"\n", b"\r\n"))
        self.file.write_bytes(self.data.replace(b"1.25", b"1.26"))
        with self.assertRaisesRegex(RuntimeError, "content mismatch"):
            verifier.verify_historical_aggregate_lineage(self.root, manifest)


if __name__ == "__main__":
    unittest.main()
