"""Sequential ablation runner: train each variant of a parameter sweep and agg results"""

from __future__ import annotations

import itertools
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.config.loader import load_config
from harness.log import get_logger

logger = get_logger(__name__)


class AblationSpec(BaseModel):
    """YAML spec for a sweep"""

    model_config = ConfigDict(extra="forbid")
    base_config: Path = Field(description="Path to base run YAML, relative to the spec file")
    base_overrides: dict[str, Any] = Field(
        default_factory=dict, description="Dot-keyed overrides applied to every variant"
    )
    sweep: dict[str, list[Any]] = Field(
        description="Dot keyed sweep dimensions, cartesian product over every list"
    )

    @model_validator(mode="after")
    def _scalars_only(self) -> AblationSpec:
        for key, value in self.base_overrides.items():
            if isinstance(value, (dict, list)):
                raise ValueError(
                    f"base_overrides['{key}'] must be a scalar, use dot-keys like 'schedule.max_steps'"
                )

        if not self.sweep:
            raise ValueError("sweep must contain at least one key")
        for key, values in self.sweep.items():
            if not values:
                raise ValueError(f"sweep['{key}'] must have at least one value")
            for v in values:
                if isinstance(v, (dict, list)):
                    raise ValueError(
                        f"sweep['{key}'] values must be scalars, got {type(v).__name__}"
                    )
        return self


def load_ablation_spec(path: Path) -> tuple[AblationSpec, Path]:
    """Parse YAML, return the validated spec and teh absolute base_config path"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ablation spec not found @{path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")

    spec = AblationSpec.model_validate(data)
    base_abs = (path.parent / spec.base_config).resolve()
    if not base_abs.exists():
        raise FileNotFoundError(
            f"base_config '{spec.base_config}' (resolved to {base_abs}) does not exist"
        )
    return spec, base_abs


@dataclass(frozen=True)
class Variant:
    """One point in the cartesian product of the sweep"""

    slug: str
    overrides: dict[str, Any]


def _format_value(value: Any) -> str:
    """Filename safe rednering of a sweep val"""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value).replace(".", "p")
    return str(value).replace(".", "p").replace("/", "_").replace(" ", "_")


def _slug_for_overrides(overrides: dict[str, Any]) -> str:
    """Stable, file name safe slug summarising"""
    parts = []
    for key, value in overrides.items():
        flat_key = key.replace(".", "_")
        parts.append(f"{flat_key}_{_format_value(value)}")
    return "__".join(parts) if parts else "default"


def generate_variants(spec: AblationSpec) -> list[Variant]:
    """Cartesian product over sweep keys"""
    keys = list(spec.sweep.keys())
    value_lists = [spec.sweep[k] for k in keys]
    variants: list[Variant] = []
    for combo in itertools.product(*value_lists):
        overrides = dict(zip(keys, combo, strict=True))
        variants.append(Variant(slug=_slug_for_overrides(overrides), overrides=overrides))
    return variants


def _overrides_to_strings(overrides: dict[str, Any]) -> list[str]:
    """flat dict -> list of key val pair"""
    out: list[str] = []
    for key, value in overrides.items():
        v = str(value).lower() if isinstance(value, bool) else str(value)
        out.append(f"{key}={v}")
    return out


@dataclass
class VariantResult:
    slug: str
    overrides: dict[str, Any]
    status: str
    final_val_loss: float | None = None
    wall_seconds: float | None = None
    error: str | None = None


def _write_variant_result(variant_dir: Path, result: VariantResult) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "result.json").write_text(json.dumps(asdict(result), indent=2))


def _read_existing_result(variant_dir: Path) -> VariantResult | None:
    path = variant_dir / "result.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return VariantResult(**data)


def _format_markdown_table(results: list[VariantResult]) -> str:
    rows = [
        "| slug | status | final_val_loss | wall_sec | overrides |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        val = f"{r.final_val_loss:.4f}" if r.final_val_loss is not None else "-"
        wall = f"{r.wall_seconds:.1f}" if r.wall_seconds is not None else "-"
        overrides_str = ", ".join(f"{k}={v}" for k, v in r.overrides.items())
        rows.append(f"| {r.slug} | {r.status} | {val} | {wall} | {overrides_str} |")
    return "\n".join(rows) + "\n"


def _write_aggregated_results(output_dir: Path, results: list[VariantResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    (output_dir / "results.md").write_text(_format_markdown_table(results))


def run_ablation(
    spec_path: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> list[VariantResult]:
    """Run every variant sequentially; aggregate to results.{json,md} under output_dir."""
    spec, base_config_abs = load_ablation_spec(Path(spec_path))
    variants = generate_variants(spec)
    output_dir = Path(output_dir).resolve()

    logger.info(
        f"ablation: {len(variants)} variants, base_config={base_config_abs}, output_dir={output_dir}"
    )
    for v in variants:
        logger.info(f"  - {v.slug}: {v.overrides}")

    if dry_run:
        logger.info("dry-run: no training will be performed")
        return [
            VariantResult(slug=v.slug, overrides=v.overrides, status="planned") for v in variants
        ]

    from harness.training.loop import train as run_train

    base_override_strings = _overrides_to_strings(spec.base_overrides)
    results: list[VariantResult] = []

    for i, variant in enumerate(variants, start=1):
        variant_dir = output_dir / variant.slug

        if not force:
            existing = _read_existing_result(variant_dir)
            if existing is not None and existing.status == "ok":
                logger.info(f"[{i}/{len(variants)}] {variant.slug}: SKIPPED (existing result.json)")
                existing.status = "skipped"
                results.append(existing)
                continue

        per_variant_overrides = (
            base_override_strings
            + _overrides_to_strings(variant.overrides)
            + [
                f"checkpoint.dir={variant_dir / 'ckpts'}",
            ]
        )

        logger.info(f"[{i}/{len(variants)}] {variant.slug}: starting")
        t0 = time.time()
        try:
            cfg = load_config(base_config_abs, per_variant_overrides)
            metrics = run_train(cfg, resume=False)
            wall = time.time() - t0
            val_loss = (metrics or {}).get("final_val_loss")
            result = VariantResult(
                slug=variant.slug,
                overrides=variant.overrides,
                status="ok",
                final_val_loss=float(val_loss) if val_loss is not None else None,
                wall_seconds=wall,
            )
            logger.info(
                f"[{i}/{len(variants)}] {variant.slug}: OK val_loss={result.final_val_loss} ({wall:.1f}s)"
            )
        except Exception as e:
            wall = time.time() - t0
            result = VariantResult(
                slug=variant.slug,
                overrides=variant.overrides,
                status="failed",
                wall_seconds=wall,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )
            logger.error(f"[{i}/{len(variants)}] {variant.slug}: FAILED ({wall:.1f}s): {e}")

        _write_variant_result(variant_dir, result)
        results.append(result)

    _write_aggregated_results(output_dir, results)
    logger.info(f"ablation complete: results in {output_dir}/results.md")
    return results
