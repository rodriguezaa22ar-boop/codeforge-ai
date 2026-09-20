"""Strict, local configuration loading for the custom cybersecurity agent.

Configuration is JSON-only on purpose: loading a policy must not execute
constructors, imports, templates, or arbitrary configuration code. The loader
also rejects unknown fields and duplicate JSON keys so a typo cannot silently
weaken a safety setting.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping

from cyber_agent.contracts import (
    ContractValidationError,
    RiskLevel,
    TransportKind,
)


class ConfigError(ContractValidationError):
    """Raised when an agent configuration is malformed or unsafe."""


_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_MAX_CONFIG_BYTES = 1024 * 1024

_TOP_LEVEL_FIELDS = {"schema_version", "agent_id", "policy", "runtime"}
_POLICY_FIELDS = {
    "max_risk",
    "allowed_transports",
    "allowed_capabilities",
    "denied_capabilities",
    "require_approval_for",
    "require_authorization_for",
    "allowed_read_roots",
    "allow_shell",
    "max_timeout_seconds",
}
_RUNTIME_FIELDS = {"request_deadline_seconds", "max_input_bytes"}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate configuration key: {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be an object")
    return value


def _check_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{label} contains unknown fields: {unknown}")


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{label} must be a positive integer")
    return value


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be an array")
    result = tuple(_string(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ConfigError(f"{label} must not contain duplicates")
    return result


def _patterns(value: Any, label: str) -> tuple[str, ...]:
    patterns = _string_list(value, label)
    for pattern in patterns:
        if len(pattern) > 128 or not any(char.isalnum() for char in pattern):
            raise ConfigError(f"{label} contains an invalid capability pattern: {pattern!r}")
        # Validate the pattern against the same grammar used for capability IDs
        # after removing glob characters. This keeps matching predictable.
        candidate = pattern.replace("*", "x").replace("?", "x")
        if not _IDENTIFIER.fullmatch(candidate):
            raise ConfigError(f"{label} contains an invalid capability pattern: {pattern!r}")
    return patterns


def _enum_list(value: Any, enum_type, label: str) -> tuple[Any, ...]:
    values = _string_list(value, label)
    try:
        return tuple(enum_type(item) for item in values)
    except ValueError as exc:
        raise ConfigError(f"{label} contains an unsupported value") from exc


def _risk(value: Any, label: str) -> RiskLevel:
    try:
        return RiskLevel(_string(value, label))
    except ValueError as exc:
        raise ConfigError(f"{label} contains an unsupported risk level") from exc


def _resolve_roots(value: Any, label: str, base_dir: Path) -> tuple[Path, ...]:
    roots = _string_list(value, label)
    resolved = tuple((base_dir / root).expanduser().resolve() for root in roots)
    if len(set(resolved)) != len(resolved):
        raise ConfigError(f"{label} must not contain duplicate paths")
    return resolved


@dataclass(frozen=True)
class PolicyConfig:
    """Policy controls applied before a capability can be invoked."""

    max_risk: RiskLevel = RiskLevel.READ_ONLY
    allowed_transports: tuple[TransportKind, ...] = (
        TransportKind.IN_PROCESS,
        TransportKind.JSONL,
    )
    allowed_capabilities: tuple[str, ...] = ("*",)
    denied_capabilities: tuple[str, ...] = ()
    require_approval_for: tuple[RiskLevel, ...] = (
        RiskLevel.WORKSPACE_WRITE,
        RiskLevel.NETWORK,
        RiskLevel.EXECUTION,
        RiskLevel.GIT_PUBLISH,
    )
    require_authorization_for: tuple[RiskLevel, ...] = (RiskLevel.NETWORK,)
    allowed_read_roots: tuple[Path, ...] = ()
    allow_shell: bool = False
    max_timeout_seconds: int = 120

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], base_dir: Path) -> "PolicyConfig":
        data = _mapping(value, "policy")
        _check_fields(data, _POLICY_FIELDS, "policy")
        kwargs: dict[str, Any] = {}
        if "max_risk" in data:
            kwargs["max_risk"] = _risk(data["max_risk"], "policy.max_risk")
        if "allowed_transports" in data:
            kwargs["allowed_transports"] = _enum_list(
                data["allowed_transports"], TransportKind, "policy.allowed_transports"
            )
        if "allowed_capabilities" in data:
            kwargs["allowed_capabilities"] = _patterns(
                data["allowed_capabilities"], "policy.allowed_capabilities"
            )
        if "denied_capabilities" in data:
            kwargs["denied_capabilities"] = _patterns(
                data["denied_capabilities"], "policy.denied_capabilities"
            )
        if "require_approval_for" in data:
            kwargs["require_approval_for"] = _enum_list(
                data["require_approval_for"], RiskLevel, "policy.require_approval_for"
            )
        if "require_authorization_for" in data:
            kwargs["require_authorization_for"] = _enum_list(
                data["require_authorization_for"], RiskLevel, "policy.require_authorization_for"
            )
        if "allowed_read_roots" in data:
            kwargs["allowed_read_roots"] = _resolve_roots(
                data["allowed_read_roots"], "policy.allowed_read_roots", base_dir
            )
        if "allow_shell" in data:
            if not isinstance(data["allow_shell"], bool):
                raise ConfigError("policy.allow_shell must be a boolean")
            kwargs["allow_shell"] = data["allow_shell"]
        if "max_timeout_seconds" in data:
            kwargs["max_timeout_seconds"] = _positive_int(
                data["max_timeout_seconds"], "policy.max_timeout_seconds"
            )
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_risk": self.max_risk.value,
            "allowed_transports": [item.value for item in self.allowed_transports],
            "allowed_capabilities": list(self.allowed_capabilities),
            "denied_capabilities": list(self.denied_capabilities),
            "require_approval_for": [item.value for item in self.require_approval_for],
            "require_authorization_for": [item.value for item in self.require_authorization_for],
            "allowed_read_roots": [str(item) for item in self.allowed_read_roots],
            "allow_shell": self.allow_shell,
            "max_timeout_seconds": self.max_timeout_seconds,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    request_deadline_seconds: int = 60
    max_input_bytes: int = 1024 * 1024

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeConfig":
        data = _mapping(value, "runtime")
        _check_fields(data, _RUNTIME_FIELDS, "runtime")
        kwargs: dict[str, Any] = {}
        if "request_deadline_seconds" in data:
            kwargs["request_deadline_seconds"] = _positive_int(
                data["request_deadline_seconds"], "runtime.request_deadline_seconds"
            )
        if "max_input_bytes" in data:
            kwargs["max_input_bytes"] = _positive_int(
                data["max_input_bytes"], "runtime.max_input_bytes"
            )
        return cls(**kwargs)

    def to_dict(self) -> dict[str, int]:
        return {
            "request_deadline_seconds": self.request_deadline_seconds,
            "max_input_bytes": self.max_input_bytes,
        }


@dataclass(frozen=True)
class AgentConfig:
    schema_version: str
    agent_id: str
    policy: PolicyConfig
    runtime: RuntimeConfig

    @classmethod
    def default(cls) -> "AgentConfig":
        return cls(
            schema_version="1.0.0",
            agent_id="cyber-agent",
            policy=PolicyConfig(),
            runtime=RuntimeConfig(),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], base_dir: Path | None = None) -> "AgentConfig":
        data = _mapping(value, "configuration")
        _check_fields(data, _TOP_LEVEL_FIELDS, "configuration")
        version = _string(data.get("schema_version"), "schema_version")
        if not _SEMVER.fullmatch(version):
            raise ConfigError(f"schema_version must use semantic version syntax: {version!r}")
        agent_id = _string(data.get("agent_id"), "agent_id")
        if not _IDENTIFIER.fullmatch(agent_id):
            raise ConfigError(f"agent_id must match lowercase identifier syntax: {agent_id!r}")
        root = (base_dir or Path.cwd()).expanduser().resolve()
        policy = PolicyConfig.from_mapping(data.get("policy", {}), root)
        runtime = RuntimeConfig.from_mapping(data.get("runtime", {}))
        if runtime.request_deadline_seconds > policy.max_timeout_seconds:
            raise ConfigError(
                "runtime.request_deadline_seconds cannot exceed policy.max_timeout_seconds"
            )
        return cls(
            schema_version=version,
            agent_id=agent_id,
            policy=policy,
            runtime=runtime,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "policy": self.policy.to_dict(),
            "runtime": self.runtime.to_dict(),
        }


class ConfigLoader:
    """Load and validate one local JSON configuration file."""

    @staticmethod
    def loads(text: str, *, base_dir: Path | None = None) -> AgentConfig:
        if not isinstance(text, str):
            raise ConfigError("configuration text must be a string")
        try:
            payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except ConfigError:
            raise
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON configuration: {exc.msg}") from exc
        return AgentConfig.from_mapping(payload, base_dir=base_dir)

    @staticmethod
    def load(path: Path) -> AgentConfig:
        candidate = Path(path).expanduser().resolve()
        try:
            if not candidate.is_file():
                raise ConfigError(f"configuration file not found: {candidate}")
            if candidate.stat().st_size > _MAX_CONFIG_BYTES:
                raise ConfigError("configuration file exceeds the 1 MiB limit")
            text = candidate.read_text(encoding="utf-8")
        except ConfigError:
            raise
        except OSError as exc:
            raise ConfigError(f"unable to read configuration: {exc}") from exc
        return ConfigLoader.loads(text, base_dir=candidate.parent)


def capability_matches(capability_id: str, patterns: tuple[str, ...]) -> bool:
    """Return whether a capability ID matches at least one configured pattern."""
    return any(fnmatchcase(capability_id, pattern) for pattern in patterns)
