import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from aind_behavior_vr_foraging_packaging.nwb_file import NwbSession
from aind_behavior_vr_foraging_packaging.pipeline import create_processors
from aind_behavior_vr_foraging_packaging.pipeline import process_session
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, ProcessStage
from aind_data_schema_models.process_names import ProcessName
from log_schema import setup_logging
from pydantic import Field
from pydantic_settings import BaseSettings

_PACKAGING_GITHUB_URL="https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.Packaging.git"
_PIPELINE_NAME = "aind-vr-foraging-pipeline"

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

def run() -> None:
    """
    Entrypoint for executing
    """
    settings = VRForagingSettings()
    start_process_time = datetime.now(tz=UTC)

    primary_data_path = tuple(settings.input_directory.glob("*"))
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
    process_name = os.getenv("PROCESS_NAME", "primary-nwb-packaging-vr-foraging")
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
        Starting acquisition nwb packaging now"
    )

    nwb_session = NwbSession(primary_data_path[0])
    processors = create_processors(nwb_session.dataset)
    nwb_session.run(*processors)

    logging.info(
        "Successfully finished nwb packaging."
    )

    logging.info(
        "Generating parquet results"
    )
    process_session(nwb_session.dataset, output_dir=settings.output_directory, processors=processors)
    logging.info(
        f"Successfully wrote parquet files to {settings.output_directory}"
    )

    nwb_result_path = settings.output_directory / "behavior.nwb.zarr"
    logging.info(f"Writing nwb to disk now at path {nwb_result_path} as zarr")
    nwb_session.write_nwb_zarr(nwb_result_path)

    end_process_time = datetime.now(tz=UTC)
    provenance = nwb_session.provenance
    data_process = DataProcess(
        start_date_time=start_process_time,
        end_date_time=end_process_time,
        stage=ProcessStage.PROCESSING,
        process_type=ProcessName.PIPELINE,
        experimenters=["Bruno Cruz", "Tiffany Ona", "Arjun Sridhar"],
        code=Code(
            url=_PACKAGING_GITHUB_URL,
            version=provenance["packaging_version"]
        ),
        output_parameters={},
        pipeline_name=_PIPELINE_NAME,
        notes=json.dumps(
            nwb_session.nwb_file.was_generated_by
        )
    )
    with open(settings.output_directory / "data_process.json", "w") as f:
        f.write(data_process.model_dump_json(indent=4))
    logging.info("Pipeline stage completed", extra={"event_type": "stage_complete"})

if __name__ == "__main__":
    try:
        run()
    except Exception:
        logging.exception("Pipeline stage failed", extra={"event_type": "stage_error"})
        raise
