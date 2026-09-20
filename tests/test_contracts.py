import json
import unittest

from cyber_agent.contracts import (
    CapabilityContract,
    CapabilityInvocation,
    CapabilityRegistry,
    CapabilityResult,
    CapabilityStatus,
    ContractValidationError,
    RiskLevel,
    TransportKind,
    TransportSpec,
)


class ContractTests(unittest.TestCase):
    def test_read_only_contract_round_trips_as_json(self):
        contract = CapabilityContract(
            id="catalog.search",
            version="0.1.0",
            purpose="Search the catalog",
            risk=RiskLevel.READ_ONLY,
            input_schema={"type": "object", "required": ["query"]},
            output_schema={"type": "object"},
        )
        payload = json.dumps(contract.to_dict())
        self.assertIn('"risk": "read_only"', payload)

    def test_jsonl_transport_requires_executable_and_disallows_shell(self):
        with self.assertRaises(ContractValidationError):
            TransportSpec(kind=TransportKind.JSONL)
        with self.assertRaises(ContractValidationError):
            TransportSpec(shell=True)

    def test_network_contract_requires_authorization_and_approval(self):
        with self.assertRaises(ContractValidationError):
            CapabilityContract(
                id="network.scan",
                version="0.1.0",
                purpose="Network scan",
                risk=RiskLevel.NETWORK,
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            )

    def test_input_schema_rejects_missing_and_wrong_values(self):
        contract = CapabilityContract(
            id="catalog.search",
            version="0.1.0",
            purpose="Search",
            risk=RiskLevel.READ_ONLY,
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string", "minLength": 1}},
            },
            output_schema={"type": "object"},
        )
        with self.assertRaises(ContractValidationError):
            contract.validate_inputs({})
        with self.assertRaises(ContractValidationError):
            contract.validate_inputs({"query": ""})

    def test_registry_rejects_duplicates_and_result_serializes(self):
        contract = CapabilityContract(
            id="repo.inspect",
            version="0.1.0",
            purpose="Inspect a repository",
            risk=RiskLevel.READ_ONLY,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        registry = CapabilityRegistry((contract,))
        with self.assertRaises(ContractValidationError):
            registry.register(contract)
        result = CapabilityResult("req-1", CapabilityStatus.COMPLETED, {"ok": True})
        self.assertEqual(result.to_dict()["status"], "completed")


if __name__ == "__main__":
    unittest.main()
