import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from aind_behavior_vr_foraging_packaging.nwb_file import NwbSession
from aind_behavior_vr_foraging_packaging.session_pipeline import create_processors
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, ProcessStage
from aind_data_schema_models.process_names import ProcessName
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

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

    logger.info(
        f"Found primary data {data_description_json['name']}. \
        Starting acquisition nwb packaging now"
    )

    nwb_session = NwbSession(primary_data_path[0])
    processors = create_processors(nwb_session.dataset)
    nwb_session.run(*processors)

    logger.info(
        "Successfully finished nwb packaging."
    )

    nwb_result_path = settings.output_directory / "behavior.nwb.zarr"
    logger.info(f"Writing to disk now at path {nwb_result_path} as zarr")
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
