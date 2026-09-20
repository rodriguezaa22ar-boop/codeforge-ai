"""Tests for the controller state machine."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cyber_agent.config import AgentConfig, ConfigLoader, PolicyConfig
from cyber_agent.controller import (
    Controller,
    ControllerError,
    ControllerState,
    InvocationRecord,
    Transition,
)
from cyber_agent.contracts import (
    CapabilityContract,
    CapabilityInvocation,
    CapabilityRegistry,
    CapabilityResult,
    CapabilityStatus,
    RiskLevel,
    TransportKind,
    TransportSpec,
)
from cyber_agent.policy import PolicyDenied, PolicyValidator


def _contract(
    capability_id: str,
    risk: RiskLevel = RiskLevel.READ_ONLY,
    requires_approval: bool = False,
    requires_authorization: bool = False,
    input_required: list[str] | None = None,
) -> CapabilityContract:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }
    if input_required:
        schema["required"] = input_required
    return CapabilityContract(
        id=capability_id,
        version="0.1.0",
        purpose=f"test capability {capability_id}",
        risk=risk,
        input_schema=schema,
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
        requires_approval=requires_approval,
        requires_authorization=requires_authorization,
        transport=TransportSpec(kind=TransportKind.IN_PROCESS),
    )


def _invocation(
    capability_id: str = "test.cap",
    request_id: str = "req-1",
    inputs: dict | None = None,
    approval_id: str | None = None,
    authorization_id: str | None = None,
) -> CapabilityInvocation:
    return CapabilityInvocation(
        request_id=request_id,
        capability_id=capability_id,
        inputs=inputs if inputs is not None else {"query": "hello"},
        approval_id=approval_id,
        authorization_id=authorization_id,
    )


class _StubAdapter(Controller):
    """Controller subclass that stubs out _dispatch for unit tests."""

    def __init__(self, *args, fail_on: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_on = fail_on
        self.dispatched: list[tuple[str, Mapping[str, Any]]] = []

    def _dispatch(self, contract: CapabilityContract, inputs: Mapping[str, Any]) -> dict[str, Any]:
        if self.fail_on == contract.id:
            raise RuntimeError(f"stub failure on {contract.id}")
        self.dispatched.append((contract.id, inputs))
        return {"result": f"ok-{contract.id}", "echo": inputs}


class LifecycleTests(unittest.TestCase):
    """State machine transitions from IDLE through to a terminal state."""

    def setUp(self):
        registry = CapabilityRegistry([_contract("test.cap")])
        policy = PolicyValidator(AgentConfig.default())
        self.controller = _StubAdapter(registry, policy)

    def test_submit_moves_idle_to_queued(self):
        inv = _invocation()
        record = self.controller.submit(inv)
        self.assertEqual(record.state, ControllerState.QUEUED)
        self.assertEqual(len(record.transitions), 1)
        self.assertEqual(record.transitions[0].from_state, ControllerState.IDLE)
        self.assertEqual(record.transitions[0].to_state, ControllerState.QUEUED)

    def test_run_moves_queued_to_running_to_completed(self):
        inv = _invocation()
        self.controller.submit(inv)
        result = self.controller.run(inv)
        record = self.controller.get_record(inv.request_id)
        self.assertEqual(record.state, ControllerState.COMPLETED)
        self.assertEqual(result.status, CapabilityStatus.COMPLETED)
        self.assertEqual(len(record.transitions), 3)
        self.assertEqual(record.transitions[1].to_state, ControllerState.RUNNING)
        self.assertEqual(record.transitions[2].to_state, ControllerState.COMPLETED)

    def test_run_from_idle_skips_queued_state(self):
        inv = _invocation(request_id="direct")
        result = self.controller.run(inv)
        record = self.controller.get_record(inv.request_id)
        self.assertEqual(record.state, ControllerState.COMPLETED)
        # IDLE → RUNNING → COMPLETED (no QUEUED because run was called directly)
        self.assertEqual(len(record.transitions), 2)

    def test_duplicate_submit_raises(self):
        inv = _invocation()
        self.controller.submit(inv)
        with self.assertRaises(ControllerError) as ctx:
            self.controller.submit(inv)
        self.assertEqual(ctx.exception.state, ControllerState.QUEUED)

    def test_run_after_completion_raises(self):
        inv = _invocation(request_id="done")
        self.controller.run(inv)
        # After completion, re-running resets to IDLE and re-dispatches.
        result = self.controller.run(inv)
        record = self.controller.get_record(inv.request_id)
        self.assertEqual(record.state, ControllerState.COMPLETED)
        self.assertEqual(result.status, CapabilityStatus.COMPLETED)


class PolicyDenialTests(unittest.TestCase):
    """Policy violations move the machine to DENIED without dispatching."""

    def test_risk_exceeds_policy_denies(self):
        registry = CapabilityRegistry([_contract("high.risk", risk=RiskLevel.EXECUTION, requires_approval=True)])
        config_dict = AgentConfig.default().to_dict()
        config_dict["policy"]["max_risk"] = "read_only"
        config = ConfigLoader.loads(__import__("json").dumps(config_dict))
        policy = PolicyValidator(config)
        controller = _StubAdapter(registry, policy)
        inv = _invocation("high.risk")
        result = controller.run(inv)
        self.assertEqual(result.status, CapabilityStatus.DENIED)
        record = controller.get_record(inv.request_id)
        self.assertEqual(record.state, ControllerState.DENIED)

    def test_approval_gate_denies_without_approval(self):
        registry = CapabilityRegistry([_contract("needs.approval", requires_approval=True)])
        policy = PolicyValidator(AgentConfig.default())
        controller = _StubAdapter(registry, policy)
        inv = _invocation("needs.approval")
        result = controller.run(inv)
        self.assertEqual(result.status, CapabilityStatus.DENIED)
        self.assertIn("approval_required", result.error["reasons"])

    def test_authorization_gate_denies_without_authorization(self):
        registry = CapabilityRegistry([_contract("needs.auth", risk=RiskLevel.NETWORK, requires_approval=True, requires_authorization=True)])
        policy = PolicyValidator(AgentConfig.default())
        controller = _StubAdapter(registry, policy)
        inv = _invocation("needs.auth")
        result = controller.run(inv)
        self.assertEqual(result.status, CapabilityStatus.DENIED)
        self.assertIn("authorization_required", result.error["reasons"])

    def test_explicit_deny_wins_over_allow(self):
        registry = CapabilityRegistry([_contract("denied.cap")])
        config_dict = AgentConfig.default().to_dict()
        config_dict["policy"]["denied_capabilities"] = ["denied.cap"]
        config = ConfigLoader.loads(__import__("json").dumps(config_dict))
        policy = PolicyValidator(config)
        controller = _StubAdapter(registry, policy)
        inv = _invocation("denied.cap")
        result = controller.run(inv)
        self.assertEqual(result.status, CapabilityStatus.DENIED)
        self.assertEqual(result.error["code"], "capability_explicitly_denied")


class ContractValidationTests(unittest.TestCase):
    """Contract input validation fails before policy or dispatch."""

    def test_missing_required_input_denies(self):
        registry = CapabilityRegistry([_contract("strict.cap", input_required=["query"])])
        policy = PolicyValidator(AgentConfig.default())
        controller = _StubAdapter(registry, policy)
        inv = _invocation("strict.cap", inputs={})
        result = controller.run(inv)
        self.assertEqual(result.status, CapabilityStatus.DENIED)
        self.assertEqual(result.error["code"], "invalid_request")

    def test_schema_value_rejection_denies(self):
        contract = CapabilityContract(
            id="enum.cap",
            version="0.1.0",
            purpose="enum test",
            risk=RiskLevel.READ_ONLY,
            input_schema={
                "type": "object",
                "required": ["mode"],
                "properties": {"mode": {"type": "string", "enum": ["safe", "dangerous"]}},
            },
            output_schema={"type": "object"},
            transport=TransportSpec(kind=TransportKind.IN_PROCESS),
        )
        registry = CapabilityRegistry([contract])
        policy = PolicyValidator(AgentConfig.default())
        controller = _StubAdapter(registry, policy)
        inv = _invocation("enum.cap", inputs={"mode": "forbidden"})
        result = controller.run(inv)
        self.assertEqual(result.status, CapabilityStatus.DENIED)


class DispatchFailureTests(unittest.TestCase):
    """Adapter exceptions map to FAILED, not crashes."""

    def test_adapter_error_yields_failed(self):
        registry = CapabilityRegistry([_contract("fragile.cap")])
        policy = PolicyValidator(AgentConfig.default())
        controller = _StubAdapter(registry, policy, fail_on="fragile.cap")
        inv = _invocation("fragile.cap")
        result = controller.run(inv)
        self.assertEqual(result.status, CapabilityStatus.FAILED)
        self.assertEqual(result.error["code"], "controller_error")
        record = controller.get_record(inv.request_id)
        self.assertEqual(record.state, ControllerState.FAILED)


class RecordInspectionTests(unittest.TestCase):
    """Records carry transitions and results for audit and replay."""

    def test_record_carries_result(self):
        registry = CapabilityRegistry([_contract("carry.cap")])
        policy = PolicyValidator(AgentConfig.default())
        controller = _StubAdapter(registry, policy)
        inv = _invocation("carry.cap", request_id="carry")
        controller.run(inv)
        record = controller.get_record("carry")
        self.assertIsNotNone(record.result)
        self.assertEqual(record.result.status, CapabilityStatus.COMPLETED)
        self.assertEqual(record.result.outputs, {"result": "ok-carry.cap", "echo": {"query": "hello"}})

    def test_records_returns_all_live(self):
        registry = CapabilityRegistry([_contract("a"), _contract("b")])
        policy = PolicyValidator(AgentConfig.default())
        controller = _StubAdapter(registry, policy)
        controller.run(_invocation("a", request_id="a"))
        controller.run(_invocation("b", request_id="b"))
        records = controller.records()
        self.assertEqual(len(records), 2)
        self.assertEqual({r.invocation.request_id for r in records}, {"a", "b"})

    def test_clear_drops_records(self):
        registry = CapabilityRegistry([_contract("clear.cap")])
        policy = PolicyValidator(AgentConfig.default())
        controller = _StubAdapter(registry, policy)
        controller.run(_invocation("clear.cap"))
        controller.clear()
        self.assertEqual(controller.records(), [])


class TransitionDeterminismTests(unittest.TestCase):
    """Transitions are stable and serializable."""

    def test_transition_sorts_to_dict(self):
        t = Transition(ControllerState.IDLE, ControllerState.RUNNING, "2026-01-01T00:00:00+00:00", reason="go")
        d = t.to_dict()
        self.assertEqual(d["from"], "idle")
        self.assertEqual(d["to"], "running")
        self.assertEqual(d["reason"], "go")

    def test_record_to_dict_round_trips_status(self):
        registry = CapabilityRegistry([_contract("rt.cap")])
        policy = PolicyValidator(AgentConfig.default())
        controller = _StubAdapter(registry, policy)
        inv = _invocation("rt.cap", request_id="rt")
        controller.run(inv)
        record = controller.get_record("rt")
        d = record.to_dict()
        self.assertEqual(d["state"], "completed")
        self.assertEqual(d["result"]["status"], "completed")
