"""Deterministic scanner adapters for repository security review.

Each scanner is a stateless, deterministic adapter that inspects a local
repository and returns findings in a standard format. Scanners are configured
explicitly and run in isolation — they never contact a network, execute a
shell, or modify files.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Finding:
    """A single security finding from a scanner."""

    category: str
    severity: str  # info | low | medium | high | critical
    file: str
    description: str
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "file": self.file,
            "description": self.description,
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class ScannerConfig:
    """Configuration for a scanner run."""

    scan_depth: int = 3
    enabled: bool = True


@dataclass(frozen=True)
class ScanResult:
    """Result of running a scanner against a repository."""

    scanner_name: str
    findings: list[Finding] = field(default_factory=list)
    categories_checked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanner": self.scanner_name,
            "findings": [f.to_dict() for f in self.findings],
            "categories_checked": self.categories_checked,
        }


class ScannerError(RuntimeError):
    """A scanner encountered a readable but non-critical problem."""


class BaseScanner:
    """Base class for deterministic repository scanners."""

    name: str = "base"
    config: ScannerConfig = ScannerConfig()

    def scan(self, repo_path: Path) -> ScanResult:
        raise NotImplementedError

    def _within_depth(self, path: Path, repo_path: Path, max_depth: int) -> bool:
        rel = path.resolve().relative_to(repo_path.resolve())
        depth = len(rel.parts) - 1 if rel != Path(".") else 0
        return depth <= max_depth

    def _is_git_dir(self, path: Path) -> bool:
        return ".git" in path.resolve().parts


class GitConfigScanner(BaseScanner):
    """Scan git configuration for security-relevant settings."""

    name = "git_config"

    def scan(self, repo_path: Path) -> ScanResult:
        findings: list[Finding] = []
        categories: set[str] = set()
        git_dir = repo_path / ".git"
        config_file = git_dir / "config"

        if not config_file.is_file():
            return ScanResult(scanner_name=self.name, categories_checked=[])

        try:
            content = config_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ScannerError(f"unable to read git config: {exc}") from exc

        # Parse git config into sections (real .git/config is INI-style)
        sections: dict[str, dict[str, str]] = {}
        current_section: str | None = None
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                sections[current_section] = {}
            elif "=" in line and current_section is not None:
                key, _, value = line.partition("=")
                sections[current_section][key.strip()] = value.strip()

        # Insecure remote URL protocols — check every [remote "..."] section
        for section_name, keys in sections.items():
            if section_name.startswith("remote") and "url" in keys:
                url = keys["url"]
                if url.startswith("http://") or url.startswith("git://"):
                    findings.append(Finding(
                        category="git_config",
                        severity="high",
                        file=str(config_file.relative_to(repo_path)),
                        description="Insecure remote URL protocol detected",
                        recommendation="Use SSH or HTTPS URLs for remote repositories",
                    ))
                    categories.add("git_config")

        # Credential store — check [credential] section
        cred_section = sections.get("credential", {})
        if cred_section.get("helper", "").lower() == "store":
            findings.append(Finding(
                category="git_config",
                severity="medium",
                file=str(config_file.relative_to(repo_path)),
                description="Git credential store enabled — credentials stored in plaintext",
                recommendation="Use credential cache or a credential manager instead of store",
            ))
            categories.add("git_config")

        # GPG signing without signing key — check [commit] and [user] sections
        commit_section = sections.get("commit", {})
        if commit_section.get("gpgsign", "false").lower() == "true":
            user_section = sections.get("user", {})
            signing_key = user_section.get("signingkey", "")
            if not signing_key:
                findings.append(Finding(
                    category="git_config",
                    severity="low",
                    file=str(config_file.relative_to(repo_path)),
                    description="GPG signing enabled but no signing key configured",
                    recommendation="Configure user.signingkey or disable commit.gpgsign if not needed",
                ))
                categories.add("git_config")

        return ScanResult(
            scanner_name=self.name,
            findings=findings,
            categories_checked=sorted(categories),
        )


class SecretsScanner(BaseScanner):
    """Scan repository files for potential secrets and sensitive data."""

    name = "secrets"

    SECRETS_PATTERNS: list[tuple[str, str, str]] = [
        (r"AKIA[0-9A-Z]{16}", "high", "AWS Access Key ID"),
        (r"AIza[0-9A-Za-z\-_]{35}", "high", "Google API Key"),
        (r"sk-[a-zA-Z0-9]{32,}", "high", "Secret Key Pattern"),
        (r"-----BEGIN RSA PRIVATE KEY-----", "critical", "Private Key (RSA)"),
        (r"-----BEGIN PRIVATE KEY-----", "critical", "Private Key"),
        (r"password\s*[:=]\s*\S+", "medium", "Password in configuration"),
        (r"api[_-]?key\s*[:=]\s*\S+", "high", "API Key in configuration"),
        (r"secret\s*[:=]\s*\S+", "high", "Secret in configuration"),
        (r"token\s*[:=]\s*\S{20,}", "high", "Token in configuration"),
    ]

    def scan(self, repo_path: Path) -> ScanResult:
        findings: list[Finding] = []
        categories: set[str] = set()

        for dirpath, dirnames, filenames in os.walk(str(repo_path)):
            dirnames.sort()
            rel_dir = Path(dirpath).relative_to(repo_path)
            depth = len(rel_dir.parts) - 1 if rel_dir != Path(".") else 0

            if depth > self.config.scan_depth:
                dirnames.clear()
                continue

            if self._is_git_dir(Path(dirpath)):
                continue

            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
                filepath = Path(dirpath) / filename
                try:
                    content = filepath.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                for pattern, severity, description in self.SECRETS_PATTERNS:
                    if re.search(pattern, content):
                        findings.append(Finding(
                            category="secrets",
                            severity=severity,
                            file=str(filepath.relative_to(repo_path)),
                            description=f"Potential {description} detected",
                            recommendation="Remove sensitive data from version control. Use environment variables or a secrets manager.",
                        ))
                        categories.add("secrets")

        return ScanResult(
            scanner_name=self.name,
            findings=findings,
            categories_checked=sorted(categories),
        )


class DependenciesScanner(BaseScanner):
    """Scan repository for dependency manifests and .env files."""

    name = "dependencies"

    MANIFEST_FILES: dict[str, str] = {
        "requirements.txt": "Python requirements — pin versions and run safety checks",
        "Pipfile": "Python Pipenv — pin versions in Pipfile.lock",
        "Pipfile.lock": "Python Pipenv lock — committed lock is good practice",
        "setup.py": "Python setup.py — pin dependencies in install_requires",
        "pyproject.toml": "Python pyproject — pin dependencies in project.dependencies",
        "package.json": "Node.js package — pin versions and review with npm audit",
        "package-lock.json": "Node.js package lock — committed lock is good practice",
        "yarn.lock": "Yarn lock — committed lock is good practice",
        "Gemfile": "Ruby Gemfile — pin versions and review with bundle audit",
        "Gemfile.lock": "Ruby Gemfile lock — committed lock is good practice",
        "composer.json": "PHP Composer — pin versions and review with composer audit",
        "composer.lock": "PHP Composer lock — committed lock is good practice",
        "Cargo.toml": "Rust Cargo — pin versions and review with cargo audit",
        "Cargo.lock": "Rust Cargo lock — committed lock is good practice",
        "go.mod": "Go module — pin versions and review with go list -m -u",
        "go.sum": "Go module sum — committed sum is good practice",
        "pom.xml": "Java Maven — pin versions and review with mvn dependency:analyze",
        "build.gradle": "Java Gradle — pin versions and review dependencies",
        "build.gradle.kts": "Java Gradle Kotlin — pin versions and review dependencies",
    }

    ENV_FILES: tuple[str, ...] = (".env", ".env.local", ".env.production", ".env.development")

    def scan(self, repo_path: Path) -> ScanResult:
        findings: list[Finding] = []
        categories: set[str] = set()

        # Scan for dependency manifests
        for dirpath, dirnames, filenames in os.walk(str(repo_path)):
            dirnames.sort()
            rel_path = Path(dirpath).relative_to(repo_path)
            depth = len(rel_path.parts) - 1 if rel_path != Path(".") else 0

            if depth > self.config.scan_depth:
                dirnames.clear()
                continue

            for manifest_name, recommendation in self.MANIFEST_FILES.items():
                if manifest_name in filenames:
                    filepath = Path(dirpath) / manifest_name
                    findings.append(Finding(
                        category="dependencies",
                        severity="info",
                        file=str(filepath.relative_to(repo_path)),
                        description=f"Dependency manifest found: {manifest_name}",
                        recommendation=recommendation,
                    ))
                    categories.add("dependencies")

        # Scan for .env files
        for dirpath, dirnames, filenames in os.walk(str(repo_path)):
            dirnames.sort()
            rel_path = Path(dirpath).relative_to(repo_path)
            depth = len(rel_path.parts) - 1 if rel_path != Path(".") else 0

            if depth > self.config.scan_depth:
                dirnames.clear()
                continue

            for env_file in self.ENV_FILES:
                if env_file in filenames:
                    filepath = Path(dirpath) / env_file
                    findings.append(Finding(
                        category="secrets",
                        severity="high",
                        file=str(filepath.relative_to(repo_path)),
                        description=f"Environment file found: {env_file} — may contain secrets",
                        recommendation="Add to .gitignore. Never commit environment files with secrets.",
                    ))
                    categories.add("secrets")

        return ScanResult(
            scanner_name=self.name,
            findings=findings,
            categories_checked=sorted(categories),
        )


class RepositorySecurityReview:
    """Orchestrates multiple deterministic scanners over a repository."""

    def __init__(
        self,
        git_config_scanner: GitConfigScanner | None = None,
        secrets_scanner: SecretsScanner | None = None,
        dependencies_scanner: DependenciesScanner | None = None,
    ) -> None:
        self.git_config_scanner = git_config_scanner or GitConfigScanner()
        self.secrets_scanner = secrets_scanner or SecretsScanner()
        self.dependencies_scanner = dependencies_scanner or DependenciesScanner()

    def review(
        self,
        repo_path: Path,
        include_git_config: bool = True,
        include_secrets: bool = True,
        include_dependencies: bool = True,
    ) -> dict[str, Any]:
        """Run all configured scanners and produce a consolidated review."""
        if not repo_path.is_dir():
            raise ScannerError(f"repository path does not exist: {repo_path}")

        repo_info: dict[str, Any] = {"path": str(repo_path), "is_git": (repo_path / ".git").is_dir()}
        all_findings: list[Finding] = []
        all_categories: set[str] = set()

        if repo_info["is_git"] and include_git_config:
            result = self.git_config_scanner.scan(repo_path)
            all_findings.extend(result.findings)
            all_categories.update(result.categories_checked)

        if include_secrets:
            result = self.secrets_scanner.scan(repo_path)
            all_findings.extend(result.findings)
            all_categories.update(result.categories_checked)

        if include_dependencies:
            result = self.dependencies_scanner.scan(repo_path)
            all_findings.extend(result.findings)
            all_categories.update(result.categories_checked)

        severity_counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for f in all_findings:
            if f.severity in severity_counts:
                severity_counts[f.severity] += 1

        return {
            "repository": repo_info,
            "findings": [f.to_dict() for f in all_findings],
            "summary": {
                "total_issues": len(all_findings),
                "by_severity": severity_counts,
                "categories_checked": sorted(all_categories),
            },
            "timestamp": self._now(),
        }

    def verify_fix(
        self,
        repo_path: Path,
        original_findings: list[dict[str, Any]],
        include_git_config: bool = True,
        include_secrets: bool = True,
        include_dependencies: bool = True,
    ) -> dict[str, Any]:
        """Re-scan a repository and compare against original findings to verify fixes.

        Returns a verification report with:
        - original: the original findings (as provided)
        - current: findings from the current scan
        - resolved: original findings no longer present
        - persistent: original findings still present
        - new: findings present now but not in the original set
        - verification_status: overall status (verified | partial | unverified)
        """
        if not repo_path.is_dir():
            raise ScannerError(f"repository path does not exist: {repo_path}")

        # Re-run scanners (current scan may skip categories the original
        # did not; findings from skipped categories are reported separately).
        current_review = self.review(
            repo_path=repo_path,
            include_git_config=include_git_config,
            include_secrets=include_secrets,
            include_dependencies=include_dependencies,
        )
        current_findings = current_review["findings"]

        # Build lookup keys for comparison: (category, file, description)
        def finding_key(f: dict[str, Any]) -> tuple[str, str, str]:
            return (
                f.get("category", "unknown"),
                f.get("file", "unknown"),
                f.get("description", "").lower().strip(),
            )

        original_keys = {finding_key(f) for f in original_findings}
        current_keys = {finding_key(f) for f in current_findings}

        # Original findings whose category was not re-scanned count as unchecked,
        # not resolved — the verifier did not look at that category.
        unchecked_categories: set[str] = set()
        if not include_git_config:
            unchecked_categories.add("git_config")
        if not include_secrets:
            unchecked_categories.add("secrets")
        if not include_dependencies:
            unchecked_categories.add("dependencies")

        unchecked_keys = {
            key for key in original_keys if key[0] in unchecked_categories
        }
        checked_original_keys = original_keys - unchecked_keys
        checked_current_keys = {
            key for key in current_keys if key[0] not in unchecked_categories
        }

        resolved_keys = checked_original_keys - checked_current_keys
        persistent_keys = checked_original_keys & checked_current_keys
        new_keys = checked_current_keys - checked_original_keys

        resolved = [f for f in original_findings if finding_key(f) in resolved_keys]
        persistent = [f for f in original_findings if finding_key(f) in persistent_keys]
        unchecked = [f for f in original_findings if finding_key(f) in unchecked_keys]
        new = [f for f in current_findings if finding_key(f) in new_keys]

        # Determine verification status
        if not original_findings:
            raise ScannerError("original_findings must be non-empty")
        if not current_findings:
            status = "verified"
        elif not persistent_keys:
            status = "verified"
        elif not resolved_keys:
            status = "unverified"
        else:
            status = "partial"

        return {
            "original": {
                "total": len(original_findings),
                "findings": original_findings,
            },
            "current": {
                "total": len(current_findings),
                "findings": current_findings,
            },
            "resolved": {
                "count": len(resolved),
                "findings": resolved,
            },
            "persistent": {
                "count": len(persistent),
                "findings": persistent,
            },
            "unchecked": {
                "count": len(unchecked),
                "findings": unchecked,
            },
            "new": {
                "count": len(new),
                "findings": new,
            },
            "verification_status": status,
            "timestamp": self._now(),
        }

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
