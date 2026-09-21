"""Evidence-backed AI triage for repository security findings.

Triage is evidence-first: findings come from deterministic scanners, and the
AI layer only adds interpretation and prioritization on top of that evidence.
When no AI provider is reachable, triage falls back to deterministic
severity-based ranking with no loss of structural information.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class TriageVerdict(str, Enum):
    """AI or deterministic triage verdict for a finding."""

    IGNORE = "ignore"
    WATCH = "watch"
    REVIEW = "review"
    ACTION_NOW = "action_now"


@dataclass(frozen=True)
class TriageNote:
    """AI or deterministic interpretation of a single finding."""

    finding_index: int
    verdict: TriageVerdict
    rationale: str
    confidence: str  # high | medium | low | deterministic

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_index": self.finding_index,
            "verdict": self.verdict.value,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TriageResult:
    """Consolidated triage output grounded in scanner evidence."""

    repository_path: str
    total_findings: int
    findings: list[dict[str, Any]]
    triage_notes: list[TriageNote]
    risk_level: str  # low | medium | high | critical
    top_priority: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_path": self.repository_path,
            "total_findings": self.total_findings,
            "findings": self.findings,
            "triage_notes": [n.to_dict() for n in self.triage_notes],
            "risk_level": self.risk_level,
            "top_priority": self.top_priority,
            "summary": self.summary,
        }


# Severity ordering for deterministic triage
_SEV_RANK = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}

# Deterministic verdict rules based on severity + category
_DETERMINISTIC_RULES: list[tuple[str, str, TriageVerdict, str]] = [
    ("critical", "secrets", TriageVerdict.ACTION_NOW,
     "Critical-severity secret detected in repository — immediate rotation required"),
    ("high", "secrets", TriageVerdict.ACTION_NOW,
     "High-severity secret pattern detected — rotate and remove from version control"),
    ("critical", "git_config", TriageVerdict.ACTION_NOW,
     "Critical git configuration issue — address before further commits"),
    ("high", "git_config", TriageVerdict.REVIEW,
     "High-severity git configuration issue — review and remediation recommended"),
    ("medium", "secrets", TriageVerdict.REVIEW,
     "Medium-severity potential secret — verify and rotate if confirmed"),
    ("medium", "git_config", TriageVerdict.WATCH,
     "Medium git configuration item — monitor and address in next maintenance window"),
    ("low", "*", TriageVerdict.WATCH,
     "Low-severity finding — track and address when convenient"),
    ("info", "*", TriageVerdict.IGNORE,
     "Informational finding — no immediate action required"),
]

# Category-based priority weighting for top_priority
_CATEGORY_WEIGHT = {
    "secrets": 100,
    "git_config": 50,
    "dependencies": 30,
}


def _deterministic_triage(findings: list[dict[str, Any]]) -> list[TriageNote]:
    """Produce deterministic triage notes for every finding."""
    notes: list[TriageNote] = []
    for idx, finding in enumerate(findings):
        severity = finding.get("severity", "info")
        category = finding.get("category", "general")
        matched = False
        for rule_sev, rule_cat, verdict, rationale in _DETERMINISTIC_RULES:
            if severity == rule_sev and (rule_cat == "*" or category == rule_cat):
                notes.append(TriageNote(
                    finding_index=idx,
                    verdict=verdict,
                    rationale=rationale,
                    confidence="deterministic",
                ))
                matched = True
                break
        if not matched:
            notes.append(TriageNote(
                finding_index=idx,
                verdict=TriageVerdict.REVIEW,
                rationale=f"Findings in category '{category}' with severity '{severity}' require manual review",
                confidence="deterministic",
            ))
    return notes


def _compute_risk_level(findings: list[dict[str, Any]], triage_notes: list[TriageNote]) -> str:
    """Compute overall risk level from findings and triage."""
    if not findings:
        return "low"

    has_action_now = any(n.verdict == TriageVerdict.ACTION_NOW for n in triage_notes)
    has_review = any(n.verdict == TriageVerdict.REVIEW for n in triage_notes)
    has_critical = any(f.get("severity") == "critical" for f in findings)
    has_high = any(f.get("severity") == "high" for f in findings)

    if has_action_now or has_critical:
        return "critical"
    if has_high:
        return "high"
    if has_review:
        return "medium"
    return "low"


def _select_top_priority(
    findings: list[dict[str, Any]],
    triage_notes: list[TriageNote],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Select top-priority findings ranked by severity then category weight."""
    ranked: list[tuple[int, dict[str, Any]]] = []
    for idx, finding in enumerate(findings):
        sev_rank = _SEV_RANK.get(finding.get("severity", "info"), 99)
        cat_weight = _CATEGORY_WEIGHT.get(finding.get("category", ""), 0)
        score = sev_rank * 1000 + (100 - cat_weight)
        ranked.append((score, finding))

    ranked.sort(key=lambda item: item[0])
    return [{"index": idx, **finding} for idx, finding in
            [(findings.index(f), f) for _, f in ranked[:limit]]]


def deterministic_triage(
    findings: list[dict[str, Any]],
    repository_path: str,
) -> TriageResult:
    """Run deterministic triage on scanner findings.

    Produces severity-based verdicts, risk level, and top-priority list
    without requiring any AI provider.
    """
    triage_notes = _deterministic_triage(findings)
    risk_level = _compute_risk_level(findings, triage_notes)
    top_priority = _select_top_priority(findings, triage_notes)

    summary = {
        "total_findings": len(findings),
        "by_verdict": {
            "action_now": sum(1 for n in triage_notes if n.verdict == TriageVerdict.ACTION_NOW),
            "review": sum(1 for n in triage_notes if n.verdict == TriageVerdict.REVIEW),
            "watch": sum(1 for n in triage_notes if n.verdict == TriageVerdict.WATCH),
            "ignore": sum(1 for n in triage_notes if n.verdict == TriageVerdict.IGNORE),
        },
        "by_severity": {
            "critical": sum(1 for f in findings if f.get("severity") == "critical"),
            "high": sum(1 for f in findings if f.get("severity") == "high"),
            "medium": sum(1 for f in findings if f.get("severity") == "medium"),
            "low": sum(1 for f in findings if f.get("severity") == "low"),
            "info": sum(1 for f in findings if f.get("severity") == "info"),
        },
        "triage_method": "deterministic",
    }

    return TriageResult(
        repository_path=repository_path,
        total_findings=len(findings),
        findings=findings,
        triage_notes=triage_notes,
        risk_level=risk_level,
        top_priority=top_priority,
        summary=summary,
    )


def ai_enrich_triage(
    triage_result: TriageResult,
    ai_prompt_fn,
    max_tokens: int = 2000,
) -> TriageResult:
    """Optionally enrich deterministic triage with AI interpretation.

    Calls ``ai_prompt_fn`` with a grounded prompt built from the findings.
    If the AI is unreachable or returns nothing, returns the original
    deterministic result unchanged. AI notes are appended with confidence
    ``medium`` and do not override deterministic verdicts.
    """
    if ai_prompt_fn is None:
        return triage_result

    prompt = _build_triage_prompt(triage_result, max_tokens)
    if not prompt:
        return triage_result

    ai_response = ai_prompt_fn(prompt)
    if not ai_response:
        return triage_result

    # Parse AI response into additional triage notes (best-effort)
    ai_notes = _parse_ai_triage_notes(ai_response, triage_result)
    if not ai_notes:
        return triage_result

    # Merge: keep deterministic notes, append AI notes for findings without
    # a deterministic verdict of ACTION_NOW (those are already clear).
    merged_notes: list[TriageNote] = []
    ai_by_index: dict[int, TriageNote] = {n.finding_index: n for n in ai_notes}
    for note in triage_result.triage_notes:
        merged_notes.append(note)
        if note.verdict != TriageVerdict.ACTION_NOW and note.finding_index in ai_by_index:
            # AI adds a second perspective; mark it as AI-derived
            ai_note = ai_by_index[note.finding_index]
            merged_notes.append(TriageNote(
                finding_index=ai_note.finding_index,
                verdict=ai_note.verdict,
                rationale=f"[AI] {ai_note.rationale}",
                confidence="medium",
            ))

    # Recompute risk level with AI input (AI can only raise, not lower)
    ai_risk = _compute_risk_level(triage_result.findings, merged_notes)
    sevs = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    combined_risk = max(triage_result.risk_level, ai_risk, key=lambda r: sevs.get(r, 0))

    return TriageResult(
        repository_path=triage_result.repository_path,
        total_findings=triage_result.total_findings,
        findings=triage_result.findings,
        triage_notes=merged_notes,
        risk_level=combined_risk,
        top_priority=triage_result.top_priority,
        summary={
            **triage_result.summary,
            "triage_method": "deterministic+ai",
            "ai_enriched": True,
        },
    )


def _build_triage_prompt(result: TriageResult, max_tokens: int) -> str:
    """Build a grounded prompt from findings for AI triage enrichment."""
    findings_text = ""
    for i, finding in enumerate(result.findings[:50]):  # cap at 50 findings
        findings_text += (
            f"\n  [{i}] {finding.get('severity', 'info').upper()} | {finding.get('category', 'general')} | "
            f"{finding.get('file', 'unknown')}: {finding.get('description', '')}\n"
            f"      Recommendation: {finding.get('recommendation', '')}"
        )

    prompt = (
        f"You are a security triage assistant. Below are {result.total_findings} findings from a "
        f"deterministic repository security scan at {result.repository_path}.\n\n"
        f"Each finding is evidence-backed (from scanners, not AI-generated). Review them and provide "
        f"triage verdicts for findings that need additional context beyond the deterministic rules.\n\n"
        f"Output format — one JSON line per finding that needs an AI verdict:\n"
        f'  {{"finding_index": 0, "verdict": "review", "rationale": "..."}}\n\n'
        f"Verdicts: ignore, watch, review, action_now\n\n"
        f"Findings (first 50):\n{findings_text}\n\n"
        f"Provide verdicts only for findings where you can add meaningful context beyond what "
        f"the deterministic rules already cover. If no findings need AI review, respond with an empty "
        f"JSON array: []\n\n"
        f"IMPORTANT: Only output the JSON array. No other text."
    )

    if len(prompt) > max_tokens:
        prompt = prompt[:max_tokens] + "\n\n[truncated — provide verdicts for visible findings only]"

    return prompt


def _parse_ai_triage_notes(
    ai_response: str,
    result: TriageResult,
) -> list[TriageNote]:
    """Best-effort parse of AI JSON array response into TriageNotes."""
    # Try to extract JSON array from response
    json_match = re.search(r"\[\s*\{.*\}\s*\]", ai_response, re.DOTALL)
    if not json_match:
        # Try line-by-line parsing
        notes: list[TriageNote] = []
        for line in ai_response.strip().splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    data = __import__("json").loads(line)
                    notes.append(TriageNote(
                        finding_index=int(data.get("finding_index", 0)),
                        verdict=TriageVerdict(data.get("verdict", "review")),
                        rationale=str(data.get("rationale", "")),
                        confidence="medium",
                    ))
                except (ValueError, KeyError, TypeError):
                    continue
        return notes

    try:
        data = __import__("json").loads(json_match.group())
    except ValueError:
        return []

    if not isinstance(data, list):
        return []

    notes: list[TriageNote] = []
    for item in data:
        try:
            notes.append(TriageNote(
                finding_index=int(item.get("finding_index", 0)),
                verdict=TriageVerdict(item.get("verdict", "review")),
                rationale=str(item.get("rationale", "")),
                confidence="medium",
            ))
        except (ValueError, KeyError, TypeError):
            continue

    return notes
