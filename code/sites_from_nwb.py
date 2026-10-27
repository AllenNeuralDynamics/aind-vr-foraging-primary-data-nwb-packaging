"""Derive sites.parquet from the NWB object (Dispatch 12, issue #40).

Only the sites table is round-tripped through NWB: the characterization in
https://github.com/AllenNeuralDynamics/aind-vr-foraging-primary-data-nwb-packaging/issues/39
showed every other table is already bit-exact with the NWB data, while
nwb.trials degrades `bool | None` columns to float64.
"""

import math
import typing as t
from pathlib import Path

import pandas as pd
from aind_behavior_vr_foraging_packaging._base import write_parquet
from aind_behavior_vr_foraging_packaging.models import Site
from hdmf_zarr import NWBZarrIO

# NWB stores missing values as NaN (e.g. `bool | None` comes back as
# 0.0 / 1.0 / NaN), so NaN is restored to None for every nullable field
_NULLABLE_FIELDS = frozenset(
    name for name, field in Site.model_fields.items() if type(None) in t.get_args(field.annotation)
)


def _restore_row(row: dict) -> dict:
    """Restore one NWB trial row to the types declared by the Site model."""
    for name in _NULLABLE_FIELDS:
        value = row[name]
        if isinstance(value, float) and math.isnan(value):
            row[name] = None
    row["odor_concentration"] = list(row["odor_concentration"])
    # validating casts every column to its declared type (e.g. 1.0 -> True)
    return Site.model_validate(row).model_dump()


def read_sites(nwb_path: Path) -> pd.DataFrame:
    """Read the sites table from nwb.trials, cast back to the Site model's types."""
    with NWBZarrIO(nwb_path.as_posix(), mode="r") as io:
        trials = io.read().trials.to_dataframe()
    rows = [_restore_row(row) for row in trials.to_dict(orient="records")]
    return pd.DataFrame(rows, columns=list(Site.model_fields))


def write_sites_parquet(nwb_path: Path, output_dir: Path, attrs: dict) -> None:
    """Write sites.parquet from the NWB object, stamped with the processor's provenance attrs."""
    sites = read_sites(nwb_path)
    sites.attrs.update(attrs)
    write_parquet(sites, output_dir / "sites.parquet")
