import json
import tempfile
import unittest
from pathlib import Path

from cyber_agent.adapters.codeforgeai_jsonl import CodeForgeAIAdapter, JsonlServer
from cyber_agent.config import PolicyConfig, RuntimeConfig
from cyber_agent.policy import PolicyValidator


class CodeForgeAIJsonlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = Path(__file__).resolve().parents[2]
        cls.codeforgeai = cls.workspace / "codeforge-ai"
        cls.server = JsonlServer(CodeForgeAIAdapter(cls.codeforgeai, [cls.codeforgeai]))

    def request(self, capability_id, inputs, request_id="test-1"):
        raw = self.server.process_line(json.dumps({
            "request_id": request_id,
            "capability_id": capability_id,
            "inputs": inputs,
        }))
        return json.loads(raw)

    def test_catalog_search_uses_real_codeforgeai_catalog(self):
        result = self.request("catalog.search", {"query": "nmap", "limit": 5})
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["outputs"]["total_matches"], 0)
        self.assertTrue(any(
            "nmap" in f'{tool["title"]} {tool["description"]}'.lower()
            for tool in result["outputs"]["tools"]
        ))

    def test_charter_is_read_only_and_structured(self):
        result = self.request("operator.charter", {})
        self.assertEqual(result["status"], "completed")
        self.assertIn("text", result["outputs"])
        self.assertIn("version", result["outputs"])

    def test_findings_parser_is_available_without_execution(self):
        raw = json.dumps({
            "template-id": "demo",
            "info": {"name": "Demo", "severity": "low"},
            "matched-at": "https://example.test",
        })
        result = self.request("findings.parse", {"parser": "nuclei", "raw": raw})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["outputs"]["findings"][0]["name"], "Demo")

    def test_unknown_capability_is_denied(self):
        result = self.request("pipeline.run", {})
        self.assertEqual(result["status"], "denied")
        self.assertEqual(result["error"]["code"], "invalid_request")

    def test_configured_policy_can_deny_a_registered_capability(self):
        policy = PolicyValidator(PolicyConfig(allowed_capabilities=("operator.*",)))
        server = JsonlServer(
            CodeForgeAIAdapter(self.codeforgeai, [self.codeforgeai], policy=policy),
            policy=policy,
        )
        raw = server.process_line(json.dumps({
            "request_id": "policy-deny-1",
            "capability_id": "catalog.search",
            "inputs": {"query": "nmap"},
        }))
        result = json.loads(raw)
        self.assertEqual(result["status"], "denied")
        self.assertEqual(result["error"]["code"], "capability_not_allowed")

    def test_runtime_input_limit_returns_structured_denial(self):
        server = JsonlServer(
            CodeForgeAIAdapter(self.codeforgeai, [self.codeforgeai]),
            runtime=RuntimeConfig(max_input_bytes=16),
        )
        result = json.loads(server.process_line('{"request_id":"too-large"}'))
        self.assertEqual(result["status"], "denied")
        self.assertEqual(result["error"]["code"], "input_too_large")

    def test_findings_load_rejects_path_outside_allowlist(self):
        with tempfile.NamedTemporaryFile(suffix=".json") as handle:
            result = self.request("findings.load", {"path": handle.name})
        self.assertEqual(result["status"], "failed")
        self.assertIn("approved read roots", result["error"]["message"])


if __name__ == "__main__":
    unittest.main()
