from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from robust_airfoil.constants import CONFIG_ROOT


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Conditions(StrictModel):
    reynolds_number: float
    mach: float
    ncrit: float


class UiucConfig(StrictModel):
    landing_url: str
    database_url: str
    zip_url: str
    individual_dir_url: str
    use_current_as_replacement: bool


class SnapshotConfig(StrictModel):
    repo_url: str
    commit: str
    case_glob: str
    geometry_dir: str
    source_tier: str


class LiveConfig(StrictModel):
    enabled: bool
    robots_url: str
    csv_url_template: str
    http_fallback_template: str
    details_url_template: str
    canary_key: str
    concurrency: int
    minimum_delay_seconds: float
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_transient_attempts: int
    max_live_pilot_records: int
    stop_on_status: list[int]
    require_contact_env: str
    permit_bulk_without_successful_canary: bool


class XfoilSourceConfig(StrictModel):
    official_page: str
    windows_zip_url: str
    documentation_url: str
    sessions_url: str
    executable_env: str
    timeout_seconds: int
    iterations: int


class SourcesConfig(StrictModel):
    project_seed: int
    conditions: Conditions
    continuity_geometry_dir: str
    legacy_dataset: str
    uiuc: UiucConfig
    snapshot: SnapshotConfig
    airfoiltools_live: LiveConfig
    xfoil: XfoilSourceConfig


class PilotConfig(StrictModel):
    target_airfoils: int = Field(ge=1)
    minimum_mapped_airfoils: int = Field(ge=1)
    minimum_joint_points: int = Field(ge=1)
    minimum_parse_success_fraction: float = Field(ge=0, le=1)
    minimum_mapping_fraction: float = Field(ge=0, le=1)
    split_counts: dict[Literal["development", "calibration", "locked_test"], int]


class ExpansionConfig(StrictModel):
    target_airfoils: int = Field(ge=1)
    start_automatically_after_go: bool


class FullConfig(StrictModel):
    build_all_available_exact_or_geometry_confirmed: bool
    start_baseline_after_expansion_pass: bool
    start_tuning_trials_after_baseline_pass: int = Field(ge=1)


class GeometryConfig(StrictModel):
    cosine_points_per_surface: int = Field(ge=21)
    exact_hash_decimals: int = Field(ge=1)
    pilot_near_duplicate_rms_threshold: float = Field(gt=0)
    minimum_local_thickness: float = Field(ge=0)


class ViabilityModelGateConfig(StrictModel):
    require_one_batch_overfit: bool
    one_batch_relative_loss_reduction: float = Field(ge=0, le=1)
    require_cl_better_than_dummy_fraction: float = Field(ge=0, le=1)
    require_log_cd_better_than_dummy_fraction: float = Field(ge=0, le=1)
    cm_improvement_is_desirable_not_initially_fatal: bool


class ViabilityConfig(StrictModel):
    seed: int
    pilot: PilotConfig
    expansion: ExpansionConfig
    full: FullConfig
    geometry: GeometryConfig
    model: ViabilityModelGateConfig
    live_network_tests: bool
    xfoil_tests: bool
    slow_tests: bool


class FeatureConfig(StrictModel):
    geometry: list[str]
    operating: list[str]


class ArchitectureConfig(StrictModel):
    type: Literal["conditioned_multi_head_mlp"]
    hidden_width: int = Field(ge=1)
    hidden_layers: int = Field(ge=1)
    activation: Literal["relu", "gelu", "silu"]
    dropout: float = Field(ge=0, lt=1)
    layer_norm: bool
    residual: bool


class TrainingConfig(StrictModel):
    optimiser: Literal["adamw"]
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    batch_size: int = Field(ge=1)
    maximum_epochs: int = Field(ge=1)
    early_stopping_patience: int = Field(ge=1)
    gradient_clip_norm: float = Field(gt=0)
    loss: Literal["huber", "logcosh", "mse"]
    huber_delta_standardised: float = Field(gt=0)
    deterministic_warn_only: bool
    mixed_precision: bool
    equalise_nominal_airfoil_weight: bool


class ModelBaselineConfig(StrictModel):
    seed: int
    features: FeatureConfig
    targets: list[Literal["cl", "log_cd", "cm"]]
    architecture: ArchitectureConfig
    training: TrainingConfig


class PrunerConfig(StrictModel):
    type: Literal["median"]
    startup_trials: int = Field(ge=0)
    warmup_epochs: int = Field(ge=0)
    interval_epochs: int = Field(ge=1)


class SearchConfig(StrictModel):
    hidden_layers: tuple[int, int]
    hidden_width_choices: list[int]
    activation_choices: list[Literal["relu", "gelu", "silu"]]
    dropout: tuple[float, float]
    learning_rate_log: tuple[float, float]
    weight_decay_log: tuple[float, float]
    batch_size_choices: list[int]
    loss_choices: list[Literal["huber", "logcosh"]]
    residual_choices: list[bool]

    @model_validator(mode="after")
    def validate_ranges(self) -> SearchConfig:
        for low, high in (self.hidden_layers, self.dropout, self.learning_rate_log, self.weight_decay_log):
            if low > high:
                raise ValueError("search range lower bound exceeds upper bound")
        return self


class TuningConfig(StrictModel):
    seed: int
    storage: str
    study_name: str
    direction: Literal["minimize"]
    n_jobs: int = Field(ge=1)
    grouped_folds: int = Field(ge=2)
    pilot_trials: int = Field(ge=1)
    automatic_initial_full_trials: int = Field(ge=1)
    pruner: PrunerConfig
    search: SearchConfig


class SampleConfig(StrictModel):
    method: Literal["scrambled_sobol"]
    count: int = Field(ge=1)
    common_random_numbers: bool | None = None
    antithetic: bool | None = None


class UncertaintyConfig(StrictModel):
    seed: int
    representation: Literal["smooth_correlated_surface_normal"]
    cosine_points_per_surface: int = Field(ge=21)
    basis_control_points_per_surface: int = Field(ge=2)
    correlation_length_chord: float = Field(gt=0)
    upper_lower_correlation: float = Field(ge=-1, le=1)
    leading_edge_taper: bool
    trailing_edge_zero_displacement: bool
    maximum_cst_refit_error_fraction_chord: float = Field(gt=0, le=0.02)
    amplitudes_fraction_chord: list[float]
    optimisation_samples: SampleConfig
    final_evaluation_samples: SampleConfig
    convergence_counts: list[int]
    ablations: list[Literal["independent_multiplicative_cst", "independent_coordinate_noise", "smooth_correlated_surface_normal"]]


class OptimisationConditions(StrictModel):
    reynolds_number: float = Field(gt=0)
    mach: float = Field(ge=0)
    ncrit: float = Field(gt=0)


class ServiceTargets(StrictModel):
    cl_required: list[float]
    weights: list[float]

    @model_validator(mode="after")
    def validate_weights(self) -> ServiceTargets:
        if len(self.cl_required) != len(self.weights) or not self.cl_required:
            raise ValueError("service targets and weights must have equal nonzero length")
        if abs(sum(self.weights) - 1.0) > 1e-9:
            raise ValueError("service target weights must sum to one")
        return self


class AvaConfig(StrictModel):
    alpha_deg: list[float]
    uncertainty_fraction: float = Field(gt=0)
    lambda_sensitivity_values: list[float]


class RelativeConstraints(StrictModel):
    thickness_ratio_lower: float
    thickness_ratio_upper: float
    section_area_lower: float
    leading_edge_radius_lower: float
    cm_allowance_below_reference: float = Field(ge=0)


class TrustConfig(StrictModel):
    require_calibrated_domain: bool
    reject_untrusted_candidates: bool


class Nsga2Config(StrictModel):
    population: int = Field(ge=4)
    generations: int = Field(ge=1)
    uncertainty_samples: int = Field(ge=1)
    independent_seeds: int | None = Field(default=None, ge=1)


class OptimisationConfig(StrictModel):
    seed: int
    reference_preference: str
    conditions: OptimisationConditions
    service_targets: ServiceTargets
    ava_baseline: AvaConfig
    risk_objectives: list[Literal["expected_weighted_cd", "cvar_95_weighted_cd"]]
    relative_constraints: RelativeConstraints
    trust: TrustConfig
    debug_nsga2: Nsga2Config
    pilot_nsga2: Nsga2Config
    full_nsga2: Nsga2Config


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def load_sources_config(path: str | Path | None = None) -> SourcesConfig:
    return SourcesConfig.model_validate(load_yaml(path or CONFIG_ROOT / "sources.yaml"))


def load_all_configs(config_root: Path = CONFIG_ROOT) -> dict[str, dict[str, Any]]:
    models: dict[str, type[StrictModel]] = {
        "sources": SourcesConfig,
        "viability": ViabilityConfig,
        "model_baseline": ModelBaselineConfig,
        "tuning": TuningConfig,
        "uncertainty": UncertaintyConfig,
        "optimisation": OptimisationConfig,
    }
    return {
        name: model.model_validate(load_yaml(config_root / f"{name}.yaml")).model_dump(mode="json")
        for name, model in models.items()
    }
