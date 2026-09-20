"""Validated, side-effect-free contracts for custom-agent capabilities.

This module defines the protocol types only. Execution belongs to an adapter or
runtime, so model output cannot turn a contract into an arbitrary shell command.
The types intentionally use the standard library and are JSON-serializable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ContractValidationError(ValueError):
    """Raised when a contract, invocation, or result violates the protocol."""


class RiskLevel(str, Enum):
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    NETWORK = "network"
    EXECUTION = "execution"
    GIT_PUBLISH = "git_publish"


class TransportKind(str, Enum):
    IN_PROCESS = "in_process"
    JSONL = "jsonl"


class CapabilityStatus(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    DENIED = "denied"


_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_RISK_ORDER = {risk: i for i, risk in enumerate(RiskLevel)}


def _id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ContractValidationError(
            f"{label} must match lowercase identifier syntax: {value!r}"
        )
    return value


def _schema(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{label} must be an object schema")
    return dict(value)


def _check_value(value: Any, schema: Mapping[str, Any], path: str = "inputs") -> None:
    """Validate the small JSON-Schema subset used at the capability boundary."""
    expected = schema.get("type")
    type_ok = {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if expected and expected in type_ok and not type_ok[expected]:
        raise ContractValidationError(f"{path} must be a {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ContractValidationError(f"{path} must be one of {schema['enum']!r}")
    if isinstance(value, str):
        minimum = schema.get("minLength")
        if minimum is not None and len(value) < minimum:
            raise ContractValidationError(f"{path} is shorter than {minimum} characters")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            raise ContractValidationError(f"{path} is below the minimum {minimum}")
        if maximum is not None and value > maximum:
            raise ContractValidationError(f"{path} is above the maximum {maximum}")
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ContractValidationError(f"{path} missing required fields: {missing}")
        properties = schema.get("properties", {})
        for key, child in properties.items():
            if key in value:
                _check_value(value[key], child, f"{path}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            _check_value(item, schema["items"], f"{path}[{index}]")


@dataclass(frozen=True)
class TransportSpec:
    kind: TransportKind = TransportKind.IN_PROCESS
    executable: str | None = None
    argv: tuple[str, ...] = ()
    shell: bool = False

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, TransportKind) else TransportKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "argv", tuple(self.argv))
        if self.shell:
            raise ContractValidationError("shell execution is forbidden by the capability protocol")
        if any(not isinstance(arg, str) or not arg for arg in self.argv):
            raise ContractValidationError("transport argv must contain non-empty strings")
        if kind is TransportKind.JSONL and not self.executable:
            raise ContractValidationError("jsonl transports require an executable")
        if kind is TransportKind.IN_PROCESS and self.executable:
            raise ContractValidationError("in_process transports cannot declare an executable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "executable": self.executable,
            "argv": list(self.argv),
            "shell": self.shell,
        }


@dataclass(frozen=True)
class CapabilityContract:
    id: str
    version: str
    purpose: str
    risk: RiskLevel
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    prerequisites: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()
    requires_authorization: bool = False
    requires_approval: bool = False
    timeout_seconds: int = 60
    transport: TransportSpec = TransportSpec()
    handler: str | None = None

    def __post_init__(self) -> None:
        _id(self.id, "capability id")
        if not isinstance(self.version, str) or not _SEMVER.fullmatch(self.version):
            raise ContractValidationError(f"version must be semantic version syntax: {self.version!r}")
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise ContractValidationError("purpose must be a non-empty string")
        risk = self.risk if isinstance(self.risk, RiskLevel) else RiskLevel(self.risk)
        object.__setattr__(self, "risk", risk)
        object.__setattr__(self, "prerequisites", tuple(self.prerequisites))
        object.__setattr__(self, "side_effects", tuple(self.side_effects))
        _schema(self.input_schema, "input_schema")
        _schema(self.output_schema, "output_schema")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, int):
            raise ContractValidationError("timeout_seconds must be an integer")
        if self.timeout_seconds <= 0:
            raise ContractValidationError("timeout_seconds must be positive")
        if risk is RiskLevel.READ_ONLY and self.side_effects:
            raise ContractValidationError("read_only capabilities cannot declare side effects")
        if risk in (RiskLevel.NETWORK, RiskLevel.EXECUTION, RiskLevel.GIT_PUBLISH):
            if not self.requires_approval:
                raise ContractValidationError(f"{risk.value} capabilities require approval")
        if risk is RiskLevel.NETWORK and not self.requires_authorization:
            raise ContractValidationError("network capabilities require authorization")

    def validate_inputs(self, inputs: Mapping[str, Any]) -> None:
        if not isinstance(inputs, Mapping):
            raise ContractValidationError("inputs must be an object")
        _check_value(inputs, self.input_schema)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "purpose": self.purpose,
            "risk": self.risk.value,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "prerequisites": list(self.prerequisites),
            "side_effects": list(self.side_effects),
            "requires_authorization": self.requires_authorization,
            "requires_approval": self.requires_approval,
            "timeout_seconds": self.timeout_seconds,
            "transport": self.transport.to_dict(),
            "handler": self.handler,
        }


@dataclass(frozen=True)
class CapabilityInvocation:
    request_id: str
    capability_id: str
    inputs: Mapping[str, Any]
    policy_context: str = "default"
    deadline_seconds: int = 60
    approval_id: str | None = None
    authorization_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ContractValidationError("request_id must be non-empty")
        _id(self.capability_id, "capability id")
        _id(self.policy_context, "policy context")
        if isinstance(self.deadline_seconds, bool) or not isinstance(self.deadline_seconds, int):
            raise ContractValidationError("deadline_seconds must be an integer")
        if self.deadline_seconds <= 0:
            raise ContractValidationError("deadline_seconds must be positive")
        if self.approval_id is not None and not isinstance(self.approval_id, str):
            raise ContractValidationError("approval_id must be a string when provided")
        if self.authorization_id is not None and not isinstance(self.authorization_id, str):
            raise ContractValidationError("authorization_id must be a string when provided")
        if self.approval_id is not None and not self.approval_id.strip():
            raise ContractValidationError("approval_id must be non-empty when provided")
        if self.authorization_id is not None and not self.authorization_id.strip():
            raise ContractValidationError("authorization_id must be non-empty when provided")

    def validate_against(self, contract: CapabilityContract) -> None:
        if self.capability_id != contract.id:
            raise ContractValidationError("invocation capability does not match contract")
        contract.validate_inputs(self.inputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "capability_id": self.capability_id,
            "inputs": dict(self.inputs),
            "policy_context": self.policy_context,
            "deadline_seconds": self.deadline_seconds,
            "approval_id": self.approval_id,
            "authorization_id": self.authorization_id,
        }


@dataclass(frozen=True)
class EvidenceRef:
    id: str
    kind: str
    source: str
    trust: str = "untrusted_tool_data"
    sha256: str | None = None
    artifact: str | None = None

    def __post_init__(self) -> None:
        _id(self.id, "evidence id")
        if not self.kind.strip() or not self.source.strip():
            raise ContractValidationError("evidence kind and source are required")
        if self.sha256 and not _SHA256.fullmatch(self.sha256):
            raise ContractValidationError("evidence sha256 must be a 64-character hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "trust": self.trust,
            "sha256": self.sha256,
            "artifact": self.artifact,
        }


@dataclass(frozen=True)
class Provenance:
    source: str
    version: str
    timestamp: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "version": self.version, "timestamp": self.timestamp}


@dataclass(frozen=True)
class CapabilityResult:
    request_id: str
    status: CapabilityStatus
    outputs: Mapping[str, Any] | None = None
    evidence_refs: tuple[EvidenceRef, ...] = ()
    warnings: tuple[str, ...] = ()
    provenance: Provenance | None = None
    error: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ContractValidationError("result request_id must be non-empty")
        status = self.status if isinstance(self.status, CapabilityStatus) else CapabilityStatus(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "outputs": dict(self.outputs or {}),
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "warnings": list(self.warnings),
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "error": dict(self.error) if self.error else None,
        }


class CapabilityRegistry:
    """In-memory allowlist of contracts; registration is deterministic."""

    def __init__(self, contracts: tuple[CapabilityContract, ...] = ()) -> None:
        self._contracts: dict[str, CapabilityContract] = {}
        for contract in contracts:
            self.register(contract)

    def register(self, contract: CapabilityContract) -> None:
        if contract.id in self._contracts:
            raise ContractValidationError(f"duplicate capability id: {contract.id}")
        self._contracts[contract.id] = contract

    def get(self, capability_id: str) -> CapabilityContract:
        try:
            return self._contracts[capability_id]
        except KeyError as exc:
            raise ContractValidationError(f"unknown capability: {capability_id}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._contracts))
