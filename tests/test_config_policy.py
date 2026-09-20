import json
import tempfile
import unittest
from pathlib import Path

from cyber_agent.config import AgentConfig, ConfigError, ConfigLoader, PolicyConfig
from cyber_agent.contracts import (
    CapabilityContract,
    CapabilityInvocation,
    RiskLevel,
    TransportKind,
    TransportSpec,
)
from cyber_agent.policy import PolicyValidator


class ConfigTests(unittest.TestCase):
    def test_loader_resolves_relative_read_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "allowed").mkdir()
            config_path = root / "agent.json"
            config_path.write_text(json.dumps({
                "schema_version": "1.0.0",
                "agent_id": "cyber-agent",
                "policy": {"allowed_read_roots": ["allowed"]},
            }), encoding="utf-8")

            config = ConfigLoader.load(config_path)

        self.assertEqual(config.agent_id, "cyber-agent")
        self.assertEqual(config.policy.allowed_read_roots, ((root / "allowed").resolve(),))

    def test_loader_rejects_unknown_fields_and_duplicate_keys(self):
        with self.assertRaises(ConfigError):
            ConfigLoader.loads(json.dumps({
                "schema_version": "1.0.0",
                "agent_id": "cyber-agent",
                "unexpected": True,
            }))
        with self.assertRaises(ConfigError):
            ConfigLoader.loads(
                '{"schema_version":"1.0.0","agent_id":"cyber-agent",'
                '"agent_id":"other-agent"}'
            )

    def test_default_config_is_read_only(self):
        config = AgentConfig.default()
        self.assertEqual(config.policy.max_risk, RiskLevel.READ_ONLY)
        self.assertFalse(config.policy.allow_shell)

    def test_runtime_deadline_cannot_exceed_policy_timeout(self):
        with self.assertRaises(ConfigError):
            ConfigLoader.loads(json.dumps({
                "schema_version": "1.0.0",
                "agent_id": "cyber-agent",
                "policy": {"max_timeout_seconds": 30},
                "runtime": {"request_deadline_seconds": 60},
            }))


class PolicyTests(unittest.TestCase):
    def read_only_contract(self, capability_id="catalog.search"):
        return CapabilityContract(
            id=capability_id,
            version="0.1.0",
            purpose="Read a capability catalog",
            risk=RiskLevel.READ_ONLY,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

    def test_allowed_capability_passes(self):
        validator = PolicyValidator(PolicyConfig(allowed_capabilities=("catalog.*",)))
        decision = validator.evaluate(self.read_only_contract())
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.code, "allowed")

    def test_explicit_deny_wins_over_allow(self):
        validator = PolicyValidator(PolicyConfig(
            allowed_capabilities=("*",),
            denied_capabilities=("catalog.*",),
        ))
        decision = validator.evaluate(self.read_only_contract())
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "capability_explicitly_denied")

    def test_network_requires_authorization_and_approval_identifiers(self):
        contract = CapabilityContract(
            id="network.scan",
            version="0.1.0",
            purpose="Perform an approved network scan",
            risk=RiskLevel.NETWORK,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            requires_authorization=True,
            requires_approval=True,
            transport=TransportSpec(kind=TransportKind.JSONL, executable="scanner"),
        )
        validator = PolicyValidator(PolicyConfig(
            max_risk=RiskLevel.NETWORK,
            allowed_capabilities=("network.*",),
            allowed_transports=(TransportKind.JSONL,),
        ))
        missing = validator.evaluate(
            contract,
            CapabilityInvocation("request-1", "network.scan", {}),
        )
        self.assertFalse(missing.allowed)
        self.assertIn("approval_required", missing.reasons)
        self.assertIn("authorization_required", missing.reasons)

        approved = validator.evaluate(
            contract,
            CapabilityInvocation(
                "request-1",
                "network.scan",
                {},
                approval_id="approval-1",
                authorization_id="authorization-1",
            ),
        )
        self.assertTrue(approved.allowed)

    def test_configured_roots_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            allowed = root / "allowed"
            outside = root / "outside"
            allowed.mkdir()
            outside.mkdir()
            validator = PolicyValidator(PolicyConfig(allowed_read_roots=(allowed,)))
            self.assertTrue(validator.validate_path(allowed / "findings.json").allowed)
            denied = validator.validate_path(outside / "findings.json")
            self.assertFalse(denied.allowed)
            self.assertEqual(denied.code, "path_outside_policy_roots")


if __name__ == "__main__":
    unittest.main()
