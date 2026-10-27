"""Write the session's parquet tables in line with Dispatch 12 (issue #40).

- sites: derived from nwb.trials. NWB can't store nulls, so `bool | None`
  columns are cast back to the Site model's types. Provenance attrs are
  restored from nwb.was_generated_by.
- every other table: bit-exact with the NWB data (see
  https://github.com/AllenNeuralDynamics/aind-vr-foraging-primary-data-nwb-packaging/issues/39),
  so each processor's write_parquet is reused to reduce duplicate logic and tech debt.
"""

import math
import typing as t
from pathlib import Path

import pandas as pd
from aind_behavior_vr_foraging_packaging._base import AbstractProcessor, write_parquet
from aind_behavior_vr_foraging_packaging._provenance import PackagingProvenance
from aind_behavior_vr_foraging_packaging.models import Site
from hdmf_zarr import NWBZarrIO

# NWB stores missing values as NaN (e.g. `bool | None` comes back as
# 0.0 / 1.0 / NaN), so NaN is restored to None for every nullable field
_NULLABLE_FIELDS = frozenset(
    name for name, field in Site.model_fields.items() if type(None) in t.get_args(field.annotation)
)


def _restore_row(row: dict) -> dict:
    """Restore one NWB trial row to the types declared by the Site model.

    Parameters
    ----------
    row : dict
        One row of ``nwb.trials``, keyed by column name.

    Returns
    -------
    dict
        The row validated and dumped by ``Site``.
    """
    for name in _NULLABLE_FIELDS:
        value = row[name]
        if isinstance(value, float) and math.isnan(value):
            row[name] = None
    row["odor_concentration"] = list(row["odor_concentration"])
    # validating casts every column to its declared type (e.g. 1.0 -> True)
    return Site.model_validate(row).model_dump()


def read_sites(nwb_path: Path) -> pd.DataFrame:
    """Read the sites table from ``nwb.trials``, cast back to the Site model's types.

    Parameters
    ----------
    nwb_path : Path
        Path to the ``behavior.nwb.zarr`` store.

    Returns
    -------
    pd.DataFrame
        One row per site, with columns in ``Site`` field order and the
        ``PackagingProvenance`` keys from ``nwb.was_generated_by`` in ``attrs``.
    """
    with NWBZarrIO(nwb_path.as_posix(), mode="r") as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        was_generated_by = dict(nwb.was_generated_by[:].tolist())
    rows = [_restore_row(row) for row in trials.to_dict(orient="records")]
    sites = pd.DataFrame(rows, columns=list(Site.model_fields))
    sites.attrs = {name: was_generated_by[name] for name in PackagingProvenance.model_fields}
    return sites


def write_parquet_files(
    processors: t.Sequence[AbstractProcessor],
    nwb_path: Path,
    output_dir: Path,
) -> None:
    """Write one parquet per processor: sites from the NWB object, the rest from the processor.

    Parameters
    ----------
    processors : sequence of AbstractProcessor
        The processors passed to ``process_session``.
    nwb_path : Path
        Path to the ``behavior.nwb.zarr`` store.
    output_dir : Path
        Directory to write the parquet files into.
    """
    for proc in processors:
        if proc.output_name == "sites":
            sites = read_sites(nwb_path)
            sites.attrs["processor"] = type(proc).__name__
            write_parquet(sites, output_dir / "sites.parquet")
        else:
            proc.write_parquet(output_dir)
