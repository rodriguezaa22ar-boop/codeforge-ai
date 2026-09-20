"""Policy evaluation for capability contracts and invocations.

The validator is deliberately separate from capability implementations. A
controller can evaluate a contract before it dispatches an adapter, and an
adapter can apply the same policy to local file paths. Denials are explicit and
carry stable codes suitable for JSONL responses and audit logs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cyber_agent.config import AgentConfig, PolicyConfig, capability_matches
from cyber_agent.contracts import (
    CapabilityContract,
    CapabilityInvocation,
    RiskLevel,
)


_RISK_ORDER = {
    RiskLevel.READ_ONLY: 0,
    RiskLevel.WORKSPACE_WRITE: 1,
    RiskLevel.NETWORK: 2,
    RiskLevel.EXECUTION: 3,
    RiskLevel.GIT_PUBLISH: 4,
}


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    code: str
    message: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "code": self.code,
            "message": self.message,
            "reasons": list(self.reasons),
        }


class PolicyDenied(Exception):
    """Raised by ``PolicyValidator.enforce`` for a denied operation."""

    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision
        super().__init__(f"{decision.code}: {decision.message}")


class PolicyValidator:
    """Evaluate contracts, invocations, and approved local paths."""

    def __init__(self, config: AgentConfig | PolicyConfig) -> None:
        self.policy = config.policy if isinstance(config, AgentConfig) else config

    def evaluate(
        self,
        contract: CapabilityContract,
        invocation: CapabilityInvocation | None = None,
    ) -> PolicyDecision:
        reasons: list[str] = []
        if not capability_matches(contract.id, self.policy.allowed_capabilities):
            reasons.append("capability_not_allowed")
        if capability_matches(contract.id, self.policy.denied_capabilities):
            reasons.append("capability_explicitly_denied")
        if _RISK_ORDER[contract.risk] > _RISK_ORDER[self.policy.max_risk]:
            reasons.append("risk_exceeds_policy")
        if contract.transport.kind not in self.policy.allowed_transports:
            reasons.append("transport_not_allowed")
        if contract.transport.shell and not self.policy.allow_shell:
            reasons.append("shell_transport_forbidden")
        if contract.timeout_seconds > self.policy.max_timeout_seconds:
            reasons.append("contract_timeout_exceeds_policy")
        if contract.risk in self.policy.require_approval_for and not contract.requires_approval:
            reasons.append("contract_missing_approval_gate")
        if (
            contract.risk in self.policy.require_authorization_for
            and not contract.requires_authorization
        ):
            reasons.append("contract_missing_authorization_gate")
        if contract.risk is RiskLevel.READ_ONLY and contract.side_effects:
            reasons.append("read_only_side_effects_declared")

        if invocation is not None:
            if invocation.deadline_seconds > self.policy.max_timeout_seconds:
                reasons.append("invocation_deadline_exceeds_policy")
            if invocation.deadline_seconds > contract.timeout_seconds:
                reasons.append("invocation_deadline_exceeds_contract")
            if (
                contract.requires_approval or contract.risk in self.policy.require_approval_for
            ) and not invocation.approval_id:
                reasons.append("approval_required")
            if (
                contract.requires_authorization
                or contract.risk in self.policy.require_authorization_for
            ) and not invocation.authorization_id:
                reasons.append("authorization_required")

        if reasons:
            return PolicyDecision(
                allowed=False,
                code=reasons[0],
                message="capability invocation denied by policy",
                reasons=tuple(reasons),
            )
        return PolicyDecision(
            allowed=True,
            code="allowed",
            message="capability invocation satisfies policy",
        )

    def enforce(
        self,
        contract: CapabilityContract,
        invocation: CapabilityInvocation | None = None,
    ) -> PolicyDecision:
        decision = self.evaluate(contract, invocation)
        if not decision.allowed:
            raise PolicyDenied(decision)
        return decision

    def validate_path(self, path: Path) -> PolicyDecision:
        """Check a path against configured roots when roots are configured.

        An empty root list means the validator adds no path restriction; the
        adapter's own approved-root allowlist remains authoritative in that
        mode. This preserves a narrow compatibility boundary for callers that
        configure roots outside the policy file.
        """
        candidate = Path(path).expanduser().resolve()
        roots = self.policy.allowed_read_roots
        if not roots:
            return PolicyDecision(True, "path_policy_not_configured", "no policy read roots configured")
        if any(candidate == root or root in candidate.parents for root in roots):
            return PolicyDecision(True, "path_allowed", "path is inside an approved policy root")
        return PolicyDecision(
            False,
            "path_outside_policy_roots",
            "path is outside the configured policy read roots",
            ("path_outside_policy_roots",),
        )
