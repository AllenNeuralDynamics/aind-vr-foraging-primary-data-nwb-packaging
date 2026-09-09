import json
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from aind_behavior_vr_foraging.data_contract import dataset as load_dataset
from aind_behavior_vr_foraging_packaging.pipeline import process_session
from aind_behavior_vr_foraging_packaging._provenance import PackagingProvenance
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.core.processing import DataProcess, ProcessStage, Processing
from aind_data_schema_models.modalities import Modality
from aind_data_schema.utils.inheritance import derive_data_description
from aind_data_schema_models.process_names import ProcessName
from log_schema import setup_logging
from pydantic import Field
from pydantic_settings import BaseSettings

_PACKAGING_GITHUB_URL="https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.Packaging.git"
_PROCESS_NAME = "primary-nwb-packaging-vr-foraging"
_FROZEN_METADATA_ASSET = "vr_paper_raw_metadata_09-02-2026"
_EXPERIMENTERS = ["Bruno Cruz", "Tiffany Ona", "Arjun Sridhar"]
_FILES_TO_COPY = [
    "acquisition",
    "instrument",
    "subject",
    "procedures"
]


class VRForagingSettings(BaseSettings, cli_parse_args=True):
    """
    Settings for VR Foraging Primary Data NWB Packaging
    """

    input_directory: Path = Field(
        default=Path("/data/"), description="Directory where data is"
    )
    output_directory: Path = Field(
        default=Path("/results/"), description="Output directory"
    )

def _read_model(metadata_dir: Path, model_cls: type, stem: str):
    """
    Reads and revalidates a frozen metadata record against the installed schema version.

    Returns None when the record is absent from the frozen asset.
    """
    record_path = metadata_dir / f"{stem}.json"
    if not record_path.exists():
        logging.warning(f"No {stem}.json found in {metadata_dir}")
        return None

    with open(record_path, "r") as f:
        record = json.load(f)

    # the frozen records carry the schema_version they were written with. Dropping it lets
    # the installed model supply its own default rather than failing on a stale literal
    record.pop("schema_version", None)
    return model_cls.model_validate(record)


def build_derived_data_description(
    metadata_dir: Path, settings: VRForagingSettings
) -> DataDescription:
    """
    Builds the derived data description for the packaged asset from the frozen raw record.
    """
    data_description_raw = _read_model(metadata_dir, DataDescription, "data_description")
    if data_description_raw is None:
        raise FileNotFoundError(f"No data_description.json found in {metadata_dir}")

    data_description_derived = derive_data_description(
        data_description_raw, process_name="processed", modalities=[Modality.BEHAVIOR]
    )
    with open(settings.output_directory / "data_description.json", "w") as f:
        f.write(data_description_derived.model_dump_json(indent=4))

    logging.info(f"Wrote derived data description {data_description_derived.name}")
    return data_description_derived


def build_processing(
    metadata_dir: Path,
    data_process: DataProcess,
    packaging_version: str,
    settings: VRForagingSettings,
) -> Processing:
    """
    Merges the upstream processing record from the frozen asset with this stage's data process
    so the derived asset carries the full provenance chain.
    """
    processing_new = Processing(
        data_processes=[data_process],
    )

    processing_raw = _read_model(metadata_dir, Processing, "processing")
    processing = processing_raw + processing_new if processing_raw else processing_new

    with open(settings.output_directory / "processing.json", "w") as f:
        f.write(processing.model_dump_json(indent=4))

    logging.info(
        f"Wrote processing with {len(processing.data_processes)} data processes"
    )
    return processing


def copy_aind_metadata(session_id: str, settings: VRForagingSettings) -> Path:
    """
    Copies the aind metadata. Since the metadata is a mix of v1/v2, there is a frozen asset
    attached with the metadata records to preserve reproducibility and have valid metadata
    """
    frozen_session_metadata_dir = settings.input_directory / _FROZEN_METADATA_ASSET / session_id
    if not frozen_session_metadata_dir.exists():
        raise FileNotFoundError(
            f"No frozen metadata found for {session_id} in {frozen_session_metadata_dir}"
        )

    for stem in _FILES_TO_COPY:
        record_path = frozen_session_metadata_dir / f"{stem}.json"
        if not record_path.exists():
            logging.warning(f"No {stem}.json found in {frozen_session_metadata_dir}")
            continue
        shutil.copy2(record_path, settings.output_directory / record_path.name)

    logging.info(f"Copied frozen metadata records for {session_id}")
    return frozen_session_metadata_dir


def run() -> None:
    """
    Entrypoint for executing
    """
    settings = VRForagingSettings()
    start_process_time = datetime.now(tz=UTC)

    primary_data_path = tuple(settings.input_directory.glob("*"))
    primary_data_path = [p for p in primary_data_path if "metadata" not in p.as_posix()]
    if not primary_data_path:
        raise FileNotFoundError("No primary data asset attached")

    if len(primary_data_path) > 1:
        raise ValueError(
            "Multiple primary data assets attached. Only single asset needed"
        )

    with open(primary_data_path[0] / "data_description.json", "r") as f:
        data_description_json = json.load(f)

    ### logging setup
    acquisition_name = data_description_json["name"]
    process_name = os.getenv("PROCESS_NAME", _PROCESS_NAME)
    pipeline_name = os.getenv("PIPELINE_NAME", "")
    setup_logging(
        (Path(__file__).parent / "logging.yml").as_posix(),
        model={
            "acquisition_name": acquisition_name,
            "process_name": process_name,
            "pipeline_name": pipeline_name    
        },
    )
    logging.info("Begin processing...", extra={"event_type": "stage_start"})
    logging.info(
        f"Found primary data {data_description_json['name']}. \
        Starting packaging now"
    )

    dataset = load_dataset(primary_data_path[0])
    process_session(
        dataset=dataset, 
        output_dir=settings.output_directory, 
        write_parquet=True,
        write_nwb=True
    )

    end_process_time = datetime.now(tz=UTC)
    logging.info(f"Succesfully wrote nwb and parquet files to {settings.output_directory}")

    provenance = PackagingProvenance.build(dataset)
    data_process = DataProcess(
        start_date_time=start_process_time,
        end_date_time=end_process_time,
        name=_PROCESS_NAME,
        stage=ProcessStage.PROCESSING,
        process_type=ProcessName.PIPELINE,
        experimenters=_EXPERIMENTERS,
        code=Code(
            url=_PACKAGING_GITHUB_URL,
            version=provenance.packaging_version
        ),
        output_parameters=provenance.model_dump(),
    )

    frozen_session_metadata_dir = copy_aind_metadata(acquisition_name, settings)
    build_derived_data_description(frozen_session_metadata_dir, settings)
    build_processing(
        frozen_session_metadata_dir,
        data_process,
        provenance.packaging_version,
        settings,
    )

    logging.info("Pipeline stage completed", extra={"event_type": "stage_complete"})

if __name__ == "__main__":
    try:
        run()
    except Exception:
        logging.exception("Pipeline stage failed", extra={"event_type": "stage_error"})
        raise
