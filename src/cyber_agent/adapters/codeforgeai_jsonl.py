"""Read-only JSONL adapter for the local CodeForgeAI capability library.

Protocol: one JSON request per stdin line, one JSON result per stdout line.
The adapter imports CodeForgeAI as a library and exposes only deterministic,
read-only operations. It never invokes a shell, installs tools, contacts a
network target, modifies repositories, or writes files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from cyber_agent.config import AgentConfig, ConfigError, ConfigLoader, RuntimeConfig
from cyber_agent.contracts import (
    CapabilityContract,
    CapabilityInvocation,
    CapabilityRegistry,
    CapabilityResult,
    CapabilityStatus,
    ContractValidationError,
    EvidenceRef,
    Provenance,
    RiskLevel,
    TransportKind,
    TransportSpec,
)
from cyber_agent.controller import Controller
from cyber_agent.policy import PolicyDenied, PolicyValidator


class AdapterError(RuntimeError):
    """A safe, user-facing adapter failure without a traceback on stdout."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any, cleaner) -> Any:
    if isinstance(value, str):
        return cleaner(value)
    if isinstance(value, list):
        return [_clean(item, cleaner) for item in value]
    if isinstance(value, dict):
        return {str(key): _clean(item, cleaner) for key, item in value.items()}
    return value


def _default_codeforgeai_root() -> Path:
    return Path(__file__).resolve().parents[4] / "codeforge-ai"


def _is_within(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _contract(
    capability_id: str,
    purpose: str,
    input_schema: Mapping[str, Any],
    output_schema: Mapping[str, Any],
    handler: str,
) -> CapabilityContract:
    return CapabilityContract(
        id=capability_id,
        version="0.1.0",
        purpose=purpose,
        risk=RiskLevel.READ_ONLY,
        input_schema=input_schema,
        output_schema=output_schema,
        transport=TransportSpec(kind=TransportKind.IN_PROCESS),
        handler=handler,
    )


def default_contracts() -> tuple[CapabilityContract, ...]:
    """Return the adapter's explicit read-only allowlist."""
    return (
        _contract(
            "catalog.search",
            "Search CodeForgeAI's approved security-tool catalog",
            {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
            {"type": "object", "required": ["tools"]},
            "catalog_search",
        ),
        _contract(
            "catalog.get",
            "Read one CodeForgeAI catalog entry by title",
            {
                "type": "object",
                "required": ["title"],
                "properties": {"title": {"type": "string", "minLength": 1}},
            },
            {"type": "object", "required": ["tool"]},
            "catalog_get",
        ),
        _contract(
            "operator.charter",
            "Read the CodeForgeAI operator safety charter",
            {"type": "object", "properties": {}},
            {"type": "object", "required": ["version", "text"]},
            "operator_charter",
        ),
        _contract(
            "findings.parse",
            "Parse previously collected tool output into normalized findings",
            {
                "type": "object",
                "required": ["parser", "raw"],
                "properties": {
                    "parser": {"type": "string", "enum": ["subfinder", "httpx", "nuclei"]},
                    "raw": {"type": "string"},
                    "timestamp": {"type": "string"},
                },
            },
            {"type": "object", "required": ["findings", "forward"]},
            "findings_parse",
        ),
        _contract(
            "findings.load",
            "Read normalized findings from an approved local path",
            {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string", "minLength": 1}},
            },
            {"type": "object", "required": ["findings"]},
            "findings_load",
        ),
        _contract(
            "report.render",
            "Render a deterministic Markdown report from approved findings",
            {
                "type": "object",
                "required": ["findings_path"],
                "properties": {
                    "findings_path": {"type": "string", "minLength": 1},
                    "name": {"type": "string"},
                    "targets": {"type": "array", "items": {"type": "string"}},
                    "scope_in": {"type": "array", "items": {"type": "string"}},
                    "scope_out": {"type": "array", "items": {"type": "string"}},
                },
            },
            {"type": "object", "required": ["markdown"]},
            "report_render",
        ),
        _contract(
            "repository.security_review",
            "Scan a local git repository for security issues and produce a structured review",
            {
                "type": "object",
                "required": ["repository_path"],
                "properties": {
                    "repository_path": {"type": "string", "minLength": 1},
                    "scan_depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                    "include_dependencies": {"type": "boolean", "default": True},
                    "include_secrets": {"type": "boolean", "default": True},
                    "include_git_config": {"type": "boolean", "default": True},
                },
            },
            {
                "type": "object",
                "required": ["repository", "findings", "summary", "timestamp"],
                "properties": {
                    "repository": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "is_git": {"type": "boolean"},
                            "branch": {"type": "string"},
                            "commit": {"type": "string"},
                            "commit_message": {"type": "string"},
                        },
                    },
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "severity": {"type": "string", "enum": ["info", "low", "medium", "high", "critical"]},
                                "file": {"type": "string"},
                                "description": {"type": "string"},
                                "recommendation": {"type": "string"},
                            },
                        },
                    },
                    "summary": {
                        "type": "object",
                        "properties": {
                            "total_issues": {"type": "integer"},
                            "by_severity": {
                                "type": "object",
                                "properties": {
                                    "critical": {"type": "integer"},
                                    "high": {"type": "integer"},
                                    "medium": {"type": "integer"},
                                    "low": {"type": "integer"},
                                    "info": {"type": "integer"},
                                },
                            },
                            "categories_checked": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "timestamp": {"type": "string"},
                },
            },
            "repository_security_review",
        ),
    )


class CodeForgeAIAdapter:
    """Lazy-loading, read-only facade over the local CodeForgeAI package."""

    def __init__(
        self,
        codeforgeai_root: Path,
        read_roots: Iterable[Path] = (),
        policy: PolicyValidator | None = None,
    ) -> None:
        self.root = codeforgeai_root.expanduser().resolve()
        self.src = self.root / "src"
        if not (self.src / "hackingtool").is_dir():
            raise AdapterError(f"CodeForgeAI source package not found under {self.src}")
        self.read_roots = tuple({self.root, *(Path(item).expanduser().resolve() for item in read_roots)})
        self.registry = CapabilityRegistry(default_contracts())
        self.policy = policy or PolicyValidator(AgentConfig.default())
        self._modules: dict[str, Any] = {}

    def _module(self, name: str):
        if name not in self._modules:
            import importlib

            if str(self.src) not in sys.path:
                sys.path.insert(0, str(self.src))
            self._modules[name] = importlib.import_module(f"hackingtool.{name}")
        return self._modules[name]

    def _safe_file(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser().resolve(strict=True)
        if not candidate.is_file():
            raise AdapterError("approved path must resolve to a regular file")
        if not _is_within(candidate, self.read_roots):
            raise AdapterError("path is outside the adapter's approved read roots")
        decision = self.policy.validate_path(candidate)
        if not decision.allowed:
            raise AdapterError(decision.message)
        if candidate.stat().st_size > 10 * 1024 * 1024:
            raise AdapterError("refusing to read files larger than 10 MiB")
        return candidate

    def dispatch(self, capability_id: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
        contract = self.registry.get(capability_id)
        contract.validate_inputs(inputs)
        return getattr(self, contract.handler)(inputs)

    def catalog_search(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        registry = self._module("registry")
        skill = self._module("skill")
        query_tokens = set(str(inputs["query"]).lower().split())
        limit = int(inputs.get("limit", 20))
        loaded = registry.load(catalog_dir=self.src / "hackingtool" / "catalog")
        matches: list[tuple[int, dict[str, Any]]] = []
        for category in loaded.categories:
            for tool in category.tools:
                fields = " ".join(
                    [tool.TITLE, tool.DESCRIPTION, " ".join(tool.TAGS), category.title]
                ).lower()
                score = sum(3 if token == tool.TITLE.lower() else 1 for token in query_tokens if token in fields)
                if score:
                    matches.append((score, {
                        "title": skill.clean(tool.TITLE),
                        "description": skill.clean(tool.DESCRIPTION),
                        "category": skill.clean(category.title),
                        "tags": [skill.clean(tag) for tag in tool.TAGS],
                        "kind": skill.clean(getattr(tool, "KIND", "install")),
                        "project_url": skill.clean(getattr(tool, "PROJECT_URL", "")),
                    }))
        matches.sort(key=lambda item: (-item[0], item[1]["title"].lower()))
        return {"tools": [item[1] for item in matches[:limit]], "total_matches": len(matches)}

    def catalog_get(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        result = self.catalog_search({"query": inputs["title"], "limit": 100})
        title = str(inputs["title"]).casefold()
        for tool in result["tools"]:
            if tool["title"].casefold() == title:
                return {"tool": tool}
        raise AdapterError(f"catalog entry not found: {inputs['title']}")

    def operator_charter(self, _inputs: Mapping[str, Any]) -> dict[str, Any]:
        skill = self._module("skill")
        return {"version": skill.version(), "text": skill.charter()}

    def findings_parse(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        findings = self._module("findings")
        skill = self._module("skill")
        parser = findings.PARSERS[inputs["parser"]]
        parsed, forward = parser(inputs["raw"], inputs.get("timestamp") or _now())
        return {
            "findings": [_clean(skill.sanitize(item), skill.clean) for item in parsed],
            "forward": [_clean(item, skill.clean) for item in forward],
        }

    def findings_load(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        findings = self._module("findings")
        skill = self._module("skill")
        path = self._safe_file(inputs["path"])
        loaded = findings.load_findings(path)
        return {"findings": [_clean(skill.sanitize(item), skill.clean) for item in loaded]}

    def report_render(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        findings = self._module("findings")
        report = self._module("report")
        skill = self._module("skill")
        path = self._safe_file(inputs["findings_path"])
        # report.render_report only reads these fields and never writes them.
        engagement = SimpleNamespace(
            name=str(inputs.get("name", "CodeForgeAI findings")),
            targets=list(inputs.get("targets", [])),
            scope_in=list(inputs.get("scope_in", [])),
            scope_out=list(inputs.get("scope_out", [])),
            created=_now(),
            findings_file=path,
        )
        # Load once here so a malformed file returns a controlled adapter error.
        findings.load_findings(path)
        return {"markdown": skill.clean(report.render_report(engagement))}

    def repository_security_review(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Scan a local git repository for security issues and produce a structured review.

        This is a read-only, local-only scan that checks:
        - Git configuration (branch protection, signing, etc.)
        - Dependency manifests for known-risk patterns
        - Secrets patterns in tracked files
        - Repository metadata and configuration
        """
        repo_path = Path(inputs["repository_path"]).expanduser().resolve()
        scan_depth = int(inputs.get("scan_depth", 3))
        include_deps = inputs.get("include_dependencies", True)
        include_secrets = inputs.get("include_secrets", True)
        include_git_config = inputs.get("include_git_config", True)

        if not repo_path.is_dir():
            raise AdapterError(f"repository path does not exist: {repo_path}")

        findings_list: list[dict[str, Any]] = []
        repo_info: dict[str, Any] = {"path": str(repo_path), "is_git": False}

        # Check if it's a git repository
        git_dir = repo_path / ".git"
        if git_dir.is_dir():
            repo_info["is_git"] = True
            if include_git_config:
                git_findings = self._scan_git_config(repo_path, scan_depth)
                findings_list.extend(git_findings)

        # Scan for secrets patterns
        if include_secrets:
            secret_findings = self._scan_for_secrets(repo_path, scan_depth)
            findings_list.extend(secret_findings)

        # Scan dependency manifests
        if include_deps:
            dep_findings = self._scan_dependencies(repo_path, scan_depth)
            findings_list.extend(dep_findings)

        # Compute summary
        severity_counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        categories: set[str] = set()
        for f in findings_list:
            sev = f.get("severity", "info")
            if sev in severity_counts:
                severity_counts[sev] += 1
            categories.add(f.get("category", "general"))

        return {
            "repository": repo_info,
            "findings": findings_list,
            "summary": {
                "total_issues": len(findings_list),
                "by_severity": severity_counts,
                "categories_checked": sorted(categories),
            },
            "timestamp": _now(),
        }

    def _scan_git_config(self, repo_path: Path, scan_depth: int) -> list[dict[str, Any]]:
        """Scan git configuration for security-relevant settings."""
        findings: list[dict[str, Any]] = []
        git_dir = repo_path / ".git"

        # Check for branch protection config (if config file exists)
        config_file = git_dir / "config"
        if config_file.is_file():
            try:
                content = config_file.read_text(encoding="utf-8", errors="replace")
                # Check for insecure protocol configurations
                if "[remote" in content:
                    for line in content.splitlines():
                        if "url = http://" in line or "url = git://" in line:
                            findings.append({
                                "category": "git_config",
                                "severity": "high",
                                "file": str(config_file.relative_to(repo_path)),
                                "description": "Insecure remote URL protocol detected",
                                "recommendation": "Use SSH or HTTPS URLs for remote repositories",
                            })
                            break
                # Check for credential helper configuration
                if "credential.helper" in content:
                    for line in content.splitlines():
                        if "credential.helper" in line and "store" in line:
                            findings.append({
                                "category": "git_config",
                                "severity": "medium",
                                "file": str(config_file.relative_to(repo_path)),
                                "description": "Git credential store enabled - credentials stored in plaintext",
                                "recommendation": "Consider using credential cache or a credential manager instead of store",
                            })
            except OSError:
                pass

        # Check for GPG signing configuration
        if config_file.is_file():
            try:
                content = config_file.read_text(encoding="utf-8", errors="replace")
                if "user.signingkey" not in content and "commit.gpgsign" in content:
                    findings.append({
                        "category": "git_config",
                        "severity": "low",
                        "file": str(config_file.relative_to(repo_path)),
                        "description": "GPG signing enabled but no signing key configured",
                        "recommendation": "Configure user.signingkey or disable commit.gpgsign if not needed",
                    })
            except OSError:
                pass

        return findings

    def _scan_for_secrets(self, repo_path: Path, scan_depth: int) -> list[dict[str, Any]]:
        """Scan repository files for potential secrets and sensitive data."""
        findings: list[dict[str, Any]] = []
        secrets_patterns = [
            (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
            (r"AIza[0-9A-Za-z\-_]{35}", "Google API Key"),
            (r"sk-[a-zA-Z0-9]{32,}", "Secret Key Pattern"),
            (r"-----BEGIN RSA PRIVATE KEY-----", "Private Key"),
            (r"-----BEGIN PRIVATE KEY-----", "Private Key"),
            (r"password\s*[:=]\s*\S+", "Password in configuration"),
            (r"api[_-]?key\s*[:=]\s*\S+", "API Key in configuration"),
            (r"secret\s*[:=]\s*\S+", "Secret in configuration"),
            (r"token\s*[:=]\s*\S{20,}", "Token in configuration"),
        ]

        for dirpath, dirnames, filenames in os.walk(str(repo_path)):
            dirnames.sort()
            rel_dir = Path(dirpath).relative_to(repo_path)
            depth = len(rel_dir.parts) if rel_dir != Path(".") else 0
            if depth > scan_depth:
                dirnames.clear()
                continue

            # Skip .git directory
            if ".git" in Path(dirpath).parts:
                continue

            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
                filepath = Path(dirpath) / filename
                try:
                    content = filepath.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                for pattern, description in secrets_patterns:
                    if pattern in content:
                        findings.append({
                            "category": "secrets",
                            "severity": "high",
                            "file": str(filepath.relative_to(repo_path)),
                            "description": f"Potential {description} detected",
                            "recommendation": "Remove sensitive data from version control. Use environment variables or a secrets manager.",
                        })
                        break  # One finding per file
        return findings

    def _scan_dependencies(self, repo_path: Path, scan_depth: int) -> list[dict[str, Any]]:
        """Scan dependency manifests for security-relevant patterns."""
        findings: list[dict[str, Any]] = []

        # Check for common dependency manifest files
        manifest_files = {
            "requirements.txt": ["requirements.txt"],
            "Pipfile": ["Pipfile"],
            "Pipfile.lock": ["Pipfile.lock"],
            "setup.py": ["setup.py"],
            "pyproject.toml": ["pyproject.toml"],
            "package.json": ["package.json"],
            "package-lock.json": ["package-lock.json"],
            "yarn.lock": ["yarn.lock"],
            "Gemfile": ["Gemfile"],
            "Gemfile.lock": ["Gemfile.lock"],
            "composer.json": ["composer.json"],
            "composer.lock": ["composer.lock"],
            "Cargo.toml": ["Cargo.toml"],
            "Cargo.lock": ["Cargo.lock"],
            "go.mod": ["go.mod"],
            "go.sum": ["go.sum"],
            "pom.xml": ["pom.xml"],
            "build.gradle": ["build.gradle"],
            "build.gradle.kts": ["build.gradle.kts"],
        }

        for manifest_name in manifest_files:
            for dirpath, dirnames, filenames in os.walk(str(repo_path)):
                dirnames.sort()
                rel_path = Path(dirpath).relative_to(repo_path)
                depth = len(rel_path.parts) if rel_path != Path(".") else 0
                if depth > scan_depth:
                    dirnames.clear()
                    continue
                if manifest_name in filenames:
                    filepath = Path(dirpath) / manifest_name
                    findings.append({
                        "category": "dependencies",
                        "severity": "info",
                        "file": str(filepath.relative_to(repo_path)),
                        "description": f"Dependency manifest found: {manifest_name}",
                        "recommendation": "Ensure all dependencies are pinned to specific versions and regularly updated. Run 'safety check' or equivalent for known vulnerabilities.",
                    })

        # Check for .env files (potential secrets)
        for dirpath, dirnames, filenames in os.walk(str(repo_path)):
            dirnames.sort()
            rel_path = Path(dirpath).relative_to(repo_path)
            depth = len(rel_path.parts) if rel_path != Path(".") else 0
            if depth > scan_depth:
                dirnames.clear()
                continue
            if ".env" in filenames or ".env.local" in filenames or ".env.production" in filenames:
                for env_file in [f for f in filenames if f.startswith(".env")]:
                    filepath = Path(dirpath) / env_file
                    findings.append({
                        "category": "secrets",
                        "severity": "high",
                        "file": str(filepath.relative_to(repo_path)),
                        "description": f"Environment file found: {env_file} - may contain secrets",
                        "recommendation": "Add .env files to .gitignore. Never commit environment files with secrets.",
                    })

        return findings


class JsonlServer:
    """Translate JSONL requests into validated, read-only capability results.

    Uses the Controller state machine for validation, policy enforcement,
    dispatch, and result recording.
    """

    def __init__(
        self,
        adapter: CodeForgeAIAdapter,
        policy: PolicyValidator | None = None,
        runtime: RuntimeConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.policy = policy or adapter.policy
        self.runtime = runtime or RuntimeConfig()
        self._controller = self._build_controller()

    def _build_controller(self) -> Controller:
        """Build a Controller wired to this adapter."""

        class _AdapterController(Controller):
            def __init__(self, adapter, registry, policy, timeout_seconds):
                super().__init__(registry, policy, timeout_seconds)
                self._adapter = adapter

            def _dispatch(self, contract: CapabilityContract, inputs: Mapping[str, Any]) -> dict[str, Any]:
                return self._adapter.dispatch(contract.id, inputs)

        return _AdapterController(self.adapter, self.adapter.registry, self.policy, self.runtime.request_deadline_seconds)

    def process_line(self, line: str) -> str | None:
        if not line.strip():
            return None
        if len(line.encode("utf-8")) > self.runtime.max_input_bytes:
            return json.dumps(CapabilityResult(
                request_id="unknown",
                status=CapabilityStatus.DENIED,
                error={
                    "code": "input_too_large",
                    "message": "request exceeds the configured input size limit",
                },
            ).to_dict(), sort_keys=True)
        request_id = None
        try:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ContractValidationError("request must be a JSON object")
            request_id = payload.get("request_id")
            invocation = CapabilityInvocation(
                request_id=str(request_id or ""),
                capability_id=payload.get("capability_id", ""),
                inputs=payload.get("inputs", {}),
                policy_context=payload.get("policy_context", "default"),
                deadline_seconds=payload.get(
                    "deadline_seconds", self.runtime.request_deadline_seconds
                ),
                approval_id=payload.get("approval_id"),
                authorization_id=payload.get("authorization_id"),
            )
            result = self._controller.run(invocation)
            result = self._controller.enrich_result(result)
        except PolicyDenied as exc:
            result = CapabilityResult(
                request_id=str(request_id or "unknown"),
                status=CapabilityStatus.DENIED,
                error={
                    "code": exc.decision.code,
                    "message": exc.decision.message,
                    "reasons": list(exc.decision.reasons),
                },
            )
        except ContractValidationError as exc:
            result = CapabilityResult(
                request_id=str(request_id or "unknown"),
                status=CapabilityStatus.DENIED,
                error={"code": "invalid_request", "message": str(exc)},
            )
        except (AdapterError, KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            result = CapabilityResult(
                request_id=str(request_id or "unknown"),
                status=CapabilityStatus.FAILED,
                error={"code": "adapter_error", "message": str(exc)},
            )
        return json.dumps(result.to_dict(), sort_keys=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codeforgeai-root", type=Path, default=None)
    parser.add_argument("--allow-read-root", action="append", type=Path, default=[])
    parser.add_argument("--config", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.codeforgeai_root or Path(
        __import__("os").environ.get("CODEFORGEAI_ROOT", _default_codeforgeai_root())
    )
    try:
        config = ConfigLoader.load(args.config) if args.config else AgentConfig.default()
        policy = PolicyValidator(config)
        read_roots = [*args.allow_read_root, *config.policy.allowed_read_roots]
        adapter = CodeForgeAIAdapter(root, read_roots, policy=policy)
        server = JsonlServer(adapter, policy=policy, runtime=config.runtime)
    except (AdapterError, ConfigError) as exc:
        print(json.dumps({"status": "failed", "error": {"code": "configuration", "message": str(exc)}}))
        return 2
    for line in sys.stdin:
        response = server.process_line(line)
        if response is not None:
            print(response, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
