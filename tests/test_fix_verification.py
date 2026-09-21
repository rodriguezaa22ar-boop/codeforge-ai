"""Tests for fix verification workflow."""

from __future__ import annotations

import unittest
import tempfile
import shutil
from pathlib import Path

from cyber_agent.adapters.scanner_adapters import RepositorySecurityReview


class FixVerificationTests(unittest.TestCase):
    """Verify that fix verification correctly classifies resolved / persistent / new findings."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fixverify-"))
        self.review = RepositorySecurityReview()
        self._seed_vulnerable_state()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _seed_vulnerable_state(self):
        """Create a repo with known vulnerable state."""
        git_dir = self.tmp / ".git" / "config"
        git_dir.parent.mkdir(parents=True, exist_ok=True)
        with git_dir.open("w") as f:
            f.write("[core]\n\trepositoryformatversion = 0\n")
            f.write('[remote "origin"]\n\turl = http://insecure.example.com/repo.git\n')
            f.write("[credential]\n\thelper = store\n")
            f.write("[commit]\n\tgpgsign = true\n")
        (self.tmp / "aws_key.txt").write_text("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n")
        (self.tmp / "config.json").write_text('{"password": "SuperSecret123"}\n')
        (self.tmp / "requirements.txt").write_text("flask==0.12.5\n")
        (self.tmp / ".env").write_text("SECRET_KEY=dev-secret\n")

    def _apply_fixes(self):
        """Simulate fixes: remove secrets, fix git remote, keep info-level items."""
        (self.tmp / "aws_key.txt").unlink()
        (self.tmp / "config.json").write_text('{"setting": "value"}\n')
        (self.tmp / ".env").unlink()
        git_dir = self.tmp / ".git" / "config"
        with git_dir.open("w") as f:
            f.write("[core]\n\trepositoryformatversion = 0\n")
            f.write('[remote "origin"]\n\turl = https://secure.example.com/repo.git\n')
            f.write("[credential]\n\thelper = cache\n")
            f.write("[commit]\n\tgpgsign = true\n")

    def test_verification_status_partial_when_info_persists(self):
        initial = self.review.review(self.tmp)
        self._apply_fixes()
        verification = self.review.verify_fix(self.tmp, initial["findings"])
        # requirements.txt (info) persists — status is partial, not verified
        self.assertEqual(verification["verification_status"], "partial")
        self.assertEqual(verification["resolved"]["count"], 3)
        self.assertEqual(verification["persistent"]["count"], 1)

    def test_resolved_findings_are_exactly_the_fixed_ones(self):
        initial = self.review.review(self.tmp)
        initial_descs = {f["description"] for f in initial["findings"]}
        self._apply_fixes()
        verification = self.review.verify_fix(self.tmp, initial["findings"])
        resolved_descs = {f["description"] for f in verification["resolved"]["findings"]}
        # The 3 resolved: insecure remote, AWS key, .env file
        self.assertIn("Insecure remote URL protocol detected", resolved_descs)
        self.assertIn("Potential AWS Access Key ID detected", resolved_descs)
        # The .env finding description includes the suffix
        self.assertTrue(
            any("Environment file found: .env" in d for d in resolved_descs),
            f"expected .env finding in {resolved_descs}",
        )

    def test_no_new_findings_when_nothing_added(self):
        initial = self.review.review(self.tmp)
        self._apply_fixes()
        verification = self.review.verify_fix(self.tmp, initial["findings"])
        self.assertEqual(verification["new"]["count"], 0)

    def test_verification_fails_on_missing_repo(self):
        initial = self.review.review(self.tmp)
        missing = self.tmp / "nonexistent"
        with self.assertRaises(Exception):
            self.review.verify_fix(missing, initial["findings"])

    def test_verification_fails_on_empty_original_findings(self):
        from cyber_agent.adapters.scanner_adapters import ScannerError
        with self.assertRaises(ScannerError):
            self.review.verify_fix(self.tmp, [])


class FixVerificationReScanTests(unittest.TestCase):
    """Verify that verify_fix re-runs scanners with the same config as review()."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fixverify-rescan-"))
        self.review = RepositorySecurityReview()
        self._seed()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _seed(self):
        git_dir = self.tmp / ".git" / "config"
        git_dir.parent.mkdir(parents=True, exist_ok=True)
        with git_dir.open("w") as f:
            f.write("[core]\n\trepositoryformatversion = 0\n")
            f.write('[remote "origin"]\n\turl = http://insecure.example.com/repo.git\n')
        (self.tmp / "secret.txt").write_text("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n")

    def test_verify_fix_includes_git_config_by_default(self):
        initial = self.review.review(self.tmp)
        self._apply_fix()
        verification = self.review.verify_fix(self.tmp, initial["findings"])
        # git_config finding should be in resolved (remote was fixed)
        resolved_categories = {f["category"] for f in verification["resolved"]["findings"]}
        self.assertIn("git_config", resolved_categories)

    def test_verify_fix_respects_include_flags(self):
        initial = self.review.review(self.tmp)
        self._apply_fix()
        # With include_git_config=False, the git_config finding goes to unchecked,
        # not resolved — the verifier did not re-scan that category.
        verification = self.review.verify_fix(
            self.tmp, initial["findings"], include_git_config=False
        )
        resolved_categories = {f["category"] for f in verification["resolved"]["findings"]}
        self.assertNotIn("git_config", resolved_categories)
        # It should appear in unchecked instead
        unchecked_categories = {f["category"] for f in verification["unchecked"]["findings"]}
        self.assertIn("git_config", unchecked_categories)

    def _apply_fix(self):
        git_dir = self.tmp / ".git" / "config"
        with git_dir.open("w") as f:
            f.write("[core]\n\trepositoryformatversion = 0\n")
            f.write('[remote "origin"]\n\turl = https://secure.example.com/repo.git\n')
        (self.tmp / "secret.txt").unlink()


if __name__ == "__main__":
    unittest.main()
