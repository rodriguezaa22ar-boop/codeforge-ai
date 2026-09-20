"""Controller state machine for the custom cybersecurity agent.

The controller owns the lifecycle of every capability invocation:

    IDLE → QUEUED → RUNNING → COMPLETED | FAILED | DENIED

It validates each request against the contract registry and policy before
dispatching to an adapter, and it never lets an adapter return a result
without first recording the state transition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from cyber_agent.contracts import (
    CapabilityContract,
    CapabilityInvocation,
    CapabilityRegistry,
    CapabilityResult,
    CapabilityStatus,
    ContractValidationError,
    Provenance,
)
from cyber_agent.policy import PolicyDenied, PolicyValidator


class ControllerState(str, Enum):
    """Lifecycle states for a single capability invocation."""

    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"


class ControllerError(RuntimeError):
    """Raised when the controller is asked to do something outside its state."""

    def __init__(self, state: ControllerState, message: str) -> None:
        self.state = state
        super().__init__(f"controller state {state.value}: {message}")


@dataclass(frozen=True)
class Transition:
    """One recorded state transition on an invocation."""

    from_state: ControllerState
    to_state: ControllerState
    timestamp: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_state.value,
            "to": self.to_state.value,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class InvocationRecord:
    """Live record for a single invocation as it moves through the machine."""

    def __init__(
        self,
        invocation: CapabilityInvocation,
        contract: CapabilityContract,
        policy: PolicyValidator,
    ) -> None:
        self.invocation = invocation
        self.contract = contract
        self.policy = policy
        self.state = ControllerState.IDLE
        self.transitions: list[Transition] = []
        self.result: CapabilityResult | None = None
        self.created_at = _now_iso()

    def transition(
        self,
        to: ControllerState,
        *,
        reason: str = "",
        result: CapabilityResult | None = None,
    ) -> Transition:
        if result is not None:
            self.result = result
        transition = Transition(
            from_state=self.state,
            to_state=to,
            timestamp=_now_iso(),
            reason=reason,
        )
        self.transitions.append(transition)
        self.state = to
        return transition

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.invocation.request_id,
            "capability_id": self.invocation.capability_id,
            "state": self.state.value,
            "transitions": [t.to_dict() for t in self.transitions],
            "created_at": self.created_at,
            "result": self.result.to_dict() if self.result else None,
        }


class Controller:
    """State machine that validates, dispatches, and records capability calls.

    Usage:

        controller = Controller(registry, adapter, policy, timeout_seconds=60)
        result = controller.run(invocation)
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        policy: PolicyValidator,
        timeout_seconds: int = 60,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.timeout_seconds = timeout_seconds
        self._records: dict[str, InvocationRecord] = {}

    def _get_or_create(self, invocation: CapabilityInvocation) -> InvocationRecord:
        if invocation.request_id in self._records:
            record = self._records[invocation.request_id]
            record.invocation = invocation
            record.contract = self.registry.get(invocation.capability_id)
            return record
        contract = self.registry.get(invocation.capability_id)
        record = InvocationRecord(invocation, contract, self.policy)
        self._records[invocation.request_id] = record
        return record

    def _deny(self, record: InvocationRecord, *, code: str, message: str, reasons: tuple[str, ...] = ()) -> CapabilityResult:
        result = CapabilityResult(
            request_id=record.invocation.request_id,
            status=CapabilityStatus.DENIED,
            error={"code": code, "message": message, "reasons": list(reasons)},
        )
        record.transition(ControllerState.DENIED, reason=f"{code}: {message}", result=result)
        return result

    def submit(self, invocation: CapabilityInvocation) -> InvocationRecord:
        """Accept an invocation into the controller's queue.

        Returns the live record. The caller can use ``run`` to dispatch it,
        or inspect the record while it waits.
        """
        record = self._get_or_create(invocation)
        if record.state is not ControllerState.IDLE:
            raise ControllerError(record.state, "invocation already in progress or finished")
        record.transition(ControllerState.QUEUED, reason="submitted to controller")
        return record

    def run(self, invocation: CapabilityInvocation) -> CapabilityResult:
        """Validate and dispatch one invocation through the full state machine.

        Transitions: IDLE/QUEUED → RUNNING → COMPLETED | FAILED | DENIED.

        Re-running a previously completed invocation resets it to IDLE first.
        """
        record = self._get_or_create(invocation)
        if record.state in (ControllerState.COMPLETED, ControllerState.FAILED, ControllerState.DENIED):
            record.transition(ControllerState.IDLE, reason="reset for re-dispatch")
        if record.state not in (ControllerState.IDLE, ControllerState.QUEUED):
            raise ControllerError(record.state, "invocation already in progress or finished")
        record.transition(ControllerState.RUNNING, reason="dispatching")

        deadline = time.monotonic() + min(
            invocation.deadline_seconds,
            self.timeout_seconds,
            record.contract.timeout_seconds,
        )

        try:
            # 1. Validate the contract itself
            record.contract.validate_inputs(invocation.inputs)

            # 2. Apply policy
            self.policy.enforce(record.contract, invocation)

            # 3. Dispatch (adapter must be wired in by subclass or external caller)
            outputs = self._dispatch(record.contract, invocation.inputs)

            result = CapabilityResult(
                request_id=invocation.request_id,
                status=CapabilityStatus.COMPLETED,
                outputs=outputs,
                provenance=Provenance("controller", "local", _now_iso()),
            )
            record.transition(ControllerState.COMPLETED, reason="dispatched successfully", result=result)
            return result

        except PolicyDenied as exc:
            return self._deny(record, code=exc.decision.code, message=exc.decision.message, reasons=exc.decision.reasons)

        except ContractValidationError as exc:
            return self._deny(record, code="invalid_request", message=str(exc))

        except Exception as exc:
            result = CapabilityResult(
                request_id=invocation.request_id,
                status=CapabilityStatus.FAILED,
                error={"code": "controller_error", "message": str(exc)},
            )
            record.transition(ControllerState.FAILED, reason=f"dispatch failed: {exc}", result=result)
            return result

    def _dispatch(self, contract: CapabilityContract, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Dispatch to the adapter for ``contract``.

        Subclasses override this to wire in a real adapter. The base
        implementation raises so the controller can still be tested without
        one.
        """
        raise NotImplementedError(f"no adapter wired for {contract.id}")

    def get_record(self, request_id: str) -> InvocationRecord | None:
        """Return the live record for a request, or ``None`` if unknown."""
        return self._records.get(request_id)

    def records(self) -> list[InvocationRecord]:
        """Return every live invocation record, in creation order."""
        return list(self._records.values())

    def clear(self) -> None:
        """Drop every record. Useful between test scenarios."""
        self._records.clear()

    def enrich_result(self, result: CapabilityResult) -> CapabilityResult:
        """Post-dispatch hook: attach controller-level metadata to a result.

        Subclasses override to add trace IDs, audit refs, or timing info.
        The default returns the result unchanged.
        """
        return result
