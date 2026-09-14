"""Load physiology YAML and require explicit provenance for every parameter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from malecns.paths import CONFIGS_DIR

ALLOWED_PROVENANCE = frozenset(
    {"MEASURED", "LITERATURE_DERIVED", "INFERRED", "ASSUMED"}
)


@dataclass(frozen=True)
class Provenanced:
    value: Any
    provenance: str
    source: str


@dataclass(frozen=True)
class Physiology:
    cv_max_extent_um: Provenanced
    cm_F_per_m2: Provenanced
    ra_ohm_cm: Provenanced
    leak_g_S_per_cm2: Provenanced
    leak_e_mV: Provenanced
    vm0_mV: Provenanced
    temperature_K: Provenanced
    spike_threshold_mV: Provenanced
    syn_tau_ms: dict[str, Provenanced]
    syn_e_mV: dict[str, Provenanced]
    unitary_weight: Provenanced
    min_delay_ms: Provenanced
    synaptic_delay_ms: Provenanced
    conduction_m_per_s: Provenanced
    residual_limit_um: Provenanced
    spike_mechanism: Provenanced


def _item(node: dict[str, Any], path: str) -> Provenanced:
    if not isinstance(node, dict) or "value" not in node or "provenance" not in node:
        raise ValueError(f"{path} must have value and provenance")
    prov = str(node["provenance"])
    if prov not in ALLOWED_PROVENANCE:
        raise ValueError(f"{path} provenance {prov!r} is not one of {sorted(ALLOWED_PROVENANCE)}")
    source = str(node.get("source", "")).strip()
    if not source:
        raise ValueError(f"{path} is missing source text")
    return Provenanced(value=node["value"], provenance=prov, source=source)


def load_physiology(path: Path | str | None = None) -> Physiology:
    path = Path(path) if path is not None else CONFIGS_DIR / "physiology.yaml"
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    syn = raw["synapses"]
    return Physiology(
        cv_max_extent_um=_item(raw["cv_policy"]["max_extent_um"], "cv_policy.max_extent_um"),
        cm_F_per_m2=_item(raw["passive"]["membrane_capacitance_F_per_m2"], "passive.cm"),
        ra_ohm_cm=_item(raw["passive"]["axial_resistivity_ohm_cm"], "passive.ra"),
        leak_g_S_per_cm2=_item(raw["passive"]["leak_conductance_S_per_cm2"], "passive.g_pas"),
        leak_e_mV=_item(raw["passive"]["leak_reversal_mV"], "passive.e_pas"),
        vm0_mV=_item(raw["passive"]["initial_voltage_mV"], "passive.Vm"),
        temperature_K=_item(raw["passive"]["temperature_K"], "passive.temp"),
        spike_threshold_mV=_item(raw["spiking"]["threshold_mV"], "spiking.threshold"),
        spike_mechanism=_item(raw["spiking"]["mechanism"], "spiking.mechanism"),
        syn_tau_ms={k: _item(v, f"syn.tau.{k}") for k, v in syn["tau_ms"].items()},
        syn_e_mV={k: _item(v, f"syn.e.{k}") for k, v in syn["reversal_mV"].items()},
        unitary_weight=_item(syn["unitary_weight"], "syn.weight"),
        min_delay_ms=_item(syn["min_delay_ms"], "syn.min_delay"),
        synaptic_delay_ms=_item(syn["synaptic_delay_ms"], "syn.delay"),
        conduction_m_per_s=_item(syn["conduction_velocity_m_per_s"], "syn.vcond"),
        residual_limit_um=_item(raw["mapping"]["residual_limit_um"], "mapping.residual"),
    )
