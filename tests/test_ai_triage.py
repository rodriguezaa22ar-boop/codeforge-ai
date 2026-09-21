"""Tests for evidence-backed AI triage."""

from __future__ import annotations

import unittest
from typing import Any

from cyber_agent.adapters.ai_triage import (
    TriageNote,
    TriageResult,
    TriageVerdict,
    _build_triage_prompt,
    _compute_risk_level,
    _deterministic_triage,
    _parse_ai_triage_notes,
    _select_top_priority,
    ai_enrich_triage,
    deterministic_triage,
)


def _finding(severity: str = "info", category: str = "general",
             **overrides) -> dict[str, Any]:
    data: dict[str, Any] = {
        "category": category,
        "severity": severity,
        "file": "test.py",
        "description": "test finding",
        "recommendation": "test recommendation",
    }
    data.update(overrides)
    return data


class DeterministicTriageTests(unittest.TestCase):
    """Deterministic triage produces verdicts without AI."""

    def test_empty_findings_low_risk(self):
        result = deterministic_triage([], ".")
        self.assertEqual(result.risk_level, "low")
        self.assertEqual(result.total_findings, 0)
        self.assertEqual(len(result.triage_notes), 0)

    def test_critical_secret_action_now(self):
        findings = [_finding("critical", "secrets", file="keys.json")]
        result = deterministic_triage(findings, ".")
        self.assertEqual(result.risk_level, "critical")
        self.assertEqual(result.triage_notes[0].verdict, TriageVerdict.ACTION_NOW)
        self.assertEqual(result.triage_notes[0].confidence, "deterministic")

    def test_high_secret_action_now(self):
        findings = [_finding("high", "secrets")]
        result = deterministic_triage(findings, ".")
        self.assertEqual(result.triage_notes[0].verdict, TriageVerdict.ACTION_NOW)

    def test_critical_git_config_action_now(self):
        findings = [_finding("critical", "git_config")]
        result = deterministic_triage(findings, ".")
        self.assertEqual(result.triage_notes[0].verdict, TriageVerdict.ACTION_NOW)

    def test_high_git_config_review(self):
        findings = [_finding("high", "git_config")]
        result = deterministic_triage(findings, ".")
        self.assertEqual(result.triage_notes[0].verdict, TriageVerdict.REVIEW)

    def test_medium_secret_review(self):
        findings = [_finding("medium", "secrets")]
        result = deterministic_triage(findings, ".")
        self.assertEqual(result.triage_notes[0].verdict, TriageVerdict.REVIEW)

    def test_low_watch(self):
        findings = [_finding("low", "dependencies")]
        result = deterministic_triage(findings, ".")
        self.assertEqual(result.triage_notes[0].verdict, TriageVerdict.WATCH)

    def test_info_ignore(self):
        findings = [_finding("info", "dependencies")]
        result = deterministic_triage(findings, ".")
        self.assertEqual(result.triage_notes[0].verdict, TriageVerdict.IGNORE)

    def test_unknown_category_review(self):
        findings = [_finding("medium", "unknown_category")]
        result = deterministic_triage(findings, ".")
        self.assertEqual(result.triage_notes[0].verdict, TriageVerdict.REVIEW)

    def test_summary_counts(self):
        findings = [
            _finding("critical", "secrets"),
            _finding("high", "git_config"),
            _finding("medium", "dependencies"),
            _finding("low", "dependencies"),
            _finding("info", "dependencies"),
        ]
        result = deterministic_triage(findings, "repo")
        self.assertEqual(result.summary["by_verdict"]["action_now"], 1)
        self.assertEqual(result.summary["by_verdict"]["review"], 2)
        self.assertEqual(result.summary["by_verdict"]["watch"], 1)
        self.assertEqual(result.summary["by_verdict"]["ignore"], 1)

    def test_triage_method_is_deterministic(self):
        result = deterministic_triage([_finding("info")], ".")
        self.assertEqual(result.summary["triage_method"], "deterministic")


class RiskLevelTests(unittest.TestCase):
    """Risk level computation from findings and triage notes."""

    def test_no_findings_is_low(self):
        self.assertEqual(_compute_risk_level([], []), "low")

    def test_action_now_is_critical(self):
        notes = [TriageNote(0, TriageVerdict.ACTION_NOW, "x", "deterministic")]
        self.assertEqual(_compute_risk_level([_finding()], notes), "critical")

    def test_critical_finding_is_critical(self):
        findings = [_finding("critical", "secrets")]
        notes = _deterministic_triage(findings)
        self.assertEqual(_compute_risk_level(findings, notes), "critical")

    def test_high_only_secrets_is_critical(self):
        findings = [_finding("high", "secrets")]
        notes = _deterministic_triage(findings)
        self.assertEqual(_compute_risk_level(findings, notes), "critical")

    def test_high_git_config_review_is_high(self):
        findings = [_finding("high", "git_config")]
        notes = _deterministic_triage(findings)
        self.assertEqual(_compute_risk_level(findings, notes), "high")

    def test_info_only_is_low(self):
        findings = [_finding("info")]
        notes = _deterministic_triage(findings)
        self.assertEqual(_compute_risk_level(findings, notes), "low")


class TopPriorityTests(unittest.TestCase):
    """Top priority selection ranks by severity then category weight."""

    def test_limited_to_five(self):
        findings = [_finding("high", "secrets") for _ in range(10)]
        top = _select_top_priority(findings, [], limit=5)
        self.assertEqual(len(top), 5)

    def sure_puts_secrets_first(self):
        findings = [
            _finding("high", "dependencies"),
            _finding("high", "secrets"),
            _finding("high", "git_config"),
        ]
        top = _select_top_priority(findings, [], limit=3)
        self.assertEqual(top[0]["category"], "secrets")
        self.assertEqual(top[1]["category"], "git_config")
        self.assertEqual(top[2]["category"], "dependencies")

    def test_critical_before_high(self):
        findings = [
            _finding("high", "secrets"),
            _finding("critical", "dependencies"),
        ]
        top = _select_top_priority(findings, [], limit=2)
        self.assertEqual(top[0]["severity"], "critical")
        self.assertEqual(top[1]["severity"], "high")

    def test_includes_index(self):
        findings = [_finding("high", "secrets")]
        top = _select_top_priority(findings, [], limit=1)
        self.assertIn("index", top[0])


class AiEnrichTests(unittest.TestCase):
    """AI enrichment appends notes without overriding deterministic ones."""

    def test_none_ai_fn_returns_original(self):
        result = deterministic_triage([_finding("info")], ".")
        enriched = ai_enrich_triage(result, None)
        self.assertEqual(len(enriched.triage_notes), 1)
        self.assertEqual(enriched.summary["triage_method"], "deterministic")

    def test_empty_ai_response_returns_original(self):
        result = deterministic_triage([_finding("info")], ".")
        enriched = ai_enrich_triage(result, lambda prompt: "")
        self.assertEqual(len(enriched.triage_notes), 1)

    def test_ai_adds_second_note(self):
        result = deterministic_triage([_finding("medium", "secrets")], ".")
        # AI provides a different perspective
        def ai_fn(prompt):
            return '[{"finding_index": 0, "verdict": "action_now", "rationale": "AI sees active usage pattern"}]'
        enriched = ai_enrich_triage(result, ai_fn)
        self.assertEqual(len(enriched.triage_notes), 2)
        self.assertEqual(enriched.triage_notes[0].verdict, TriageVerdict.REVIEW)  # deterministic
        self.assertEqual(enriched.triage_notes[1].verdict, TriageVerdict.ACTION_NOW)  # AI
        self.assertTrue(enriched.triage_notes[1].rationale.startswith("[AI]"))

    def test_ai_can_raise_risk(self):
        findings = [_finding("low", "dependencies")]
        result = deterministic_triage(findings, ".")
        self.assertEqual(result.risk_level, "low")
        def ai_fn(prompt):
            return '[{"finding_index": 0, "verdict": "action_now", "rationale": "AI: actively exploited"}]'
        enriched = ai_enrich_triage(result, ai_fn)
        self.assertEqual(enriched.risk_level, "critical")

    def test_ai_cannot_lower_risk(self):
        findings = [_finding("critical", "secrets")]
        result = deterministic_triage(findings, ".")
        self.assertEqual(result.risk_level, "critical")
        def ai_fn(prompt):
            return '[{"finding_index": 0, "verdict": "ignore", "rationale": "False positive"}]'
        enriched = ai_enrich_triage(result, ai_fn)
        # AI said ignore, but deterministic says action_now → stays critical
        self.assertEqual(enriched.risk_level, "critical")

    def test_ai_enriched_method_flag(self):
        result = deterministic_triage([_finding("info")], ".")
        def ai_fn(prompt):
            return '[{"finding_index": 0, "verdict": "watch", "rationale": "AI note"}]'
        enriched = ai_enrich_triage(result, ai_fn)
        self.assertEqual(enriched.summary["triage_method"], "deterministic+ai")
        self.assertTrue(enriched.summary["ai_enriched"])


class PromptBuildingTests(unittest.TestCase):
    """Prompt construction for AI enrichment."""

    def test_prompt_includes_findings(self):
        findings = [_finding("high", "secrets", description="AWS key leaked")]
        result = deterministic_triage(findings, "/repo")
        prompt = _build_triage_prompt(result, 4096)
        self.assertIn("/repo", prompt)
        self.assertIn("AWS key leaked", prompt)
        self.assertIn("HIGH", prompt)

    def test_prompt_truncates_long_findings(self):
        findings = [_finding("info", "d", description="x" * 1000) for _ in range(60)]
        result = deterministic_triage(findings, "/r")
        prompt = _build_triage_prompt(result, 500)
        self.assertIn("[truncated", prompt)

    def test_empty_findings_prompt(self):
        result = deterministic_triage([], "/r")
        prompt = _build_triage_prompt(result, 4096)
        self.assertIn("0 findings", prompt)


class AiNoteParsingTests(unittest.TestCase):
    """Parsing of AI JSON responses."""

    def test_parse_json_array(self):
        response = '[{"finding_index": 0, "verdict": "review", "rationale": "needs context"}]'
        notes = _parse_ai_triage_notes(response, TriageResult(
            repository_path="", total_findings=1, findings=[], triage_notes=[],
            risk_level="low", top_priority=[], summary={},
        ))
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].verdict, TriageVerdict.REVIEW)

    def test_parse_line_by_line(self):
        response = '{"finding_index": 1, "verdict": "watch", "rationale": "line1"}'
        notes = _parse_ai_triage_notes(response, TriageResult(
            repository_path="", total_findings=2, findings=[{}, {}], triage_notes=[],
            risk_level="low", top_priority=[], summary={},
        ))
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].finding_index, 1)

    def test_empty_array(self):
        notes = _parse_ai_triage_notes("[]", TriageResult(
            repository_path="", total_findings=0, findings=[], triage_notes=[],
            risk_level="low", top_priority=[], summary={},
        ))
        self.assertEqual(notes, [])

    def test_invalid_json_returns_empty(self):
        notes = _parse_ai_triage_notes("not json at all", TriageResult(
            repository_path="", total_findings=0, findings=[], triage_notes=[],
            risk_level="low", top_priority=[], summary={},
        ))
        self.assertEqual(notes, [])

    def test_missing_fields_default_safely(self):
        notes = _parse_ai_triage_notes(' [{"foo": "bar"}] ', TriageResult(
            repository_path="", total_findings=0, findings=[], triage_notes=[],
            risk_level="low", top_priority=[], summary={},
        ))
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].verdict, TriageVerdict.REVIEW)  # default
        self.assertEqual(notes[0].finding_index, 0)  # default
