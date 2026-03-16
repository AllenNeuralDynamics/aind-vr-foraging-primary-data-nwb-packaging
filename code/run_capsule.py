import json
import logging
from datetime import datetime
from pathlib import Path

import contraqctor.contract as data_contract
from contraqctor.contract.base import DataStream
import pynwb

from aind_data_access_api.document_db import MetadataDbClient
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, ProcessStage
from aind_data_schema_models.process_names import ProcessName
from aind_nwb_utils.utils import create_base_nwb_file
from dateutil import parser
from hdmf_zarr import NWBZarrIO
from ndx_events import NdxEventsNWBFile
from pydantic import Field
from pydantic_core import ValidationError
from pydantic_settings import BaseSettings
from pynwb import TimeSeries
from pynwb.base import ProcessingModule

from models import Site
from processing import DatasetProcessor

import utils

logger = logging.getLogger(__name__)
API_GATEWAY_HOST = "api.allenneuraldynamics.org"
VERSION="8.0"
GITHUB_URL="https://github.com/AllenNeuralDynamics/aind-vr-foraging-primary-data-nwb-packaging.git"

import json
from pathlib import Path


def write_ancillary_metadata(response: dict, path: str | Path) -> None:
    """
    Writes ancillary metadata from a response dict to a specified directory.
    Each key is written as its own JSON file.

    Parameters
    ----------
    response : dict
        Dictionary containing the keys: acquisition, data_description,
        procedures, instrument, and subject.
    path : str or Path
        Directory where the JSON files will be written.
    """
    keys = ["acquisition", "data_description", "procedures", "instrument", "subject"]
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)

    for key in keys:
        if key not in response:
            print(f"Warning: '{key}' not found in response, skipping.")
            continue

        file_path = output_path / f"{key}.json"
        with open(file_path, "w") as f:
            json.dump(response[key], f, indent=4)

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

    docdb_api_client = MetadataDbClient(
        host=API_GATEWAY_HOST,
        version="v2"
    )


    settings = VRForagingSettings()
    start_process_time = datetime.now()

    primary_data_path = tuple(settings.input_directory.glob("*"))
    if not primary_data_path:
        raise FileNotFoundError("No primary data asset attached")

    if len(primary_data_path) > 1:
        raise ValueError(
            "Multiple primary data assets attached. Only single asset needed"
        )

    #acquisition_json_path = tuple(settings.input_directory.glob("*/acquisition.json"))
    data_description_json_path = tuple(
        settings.input_directory.glob("*/data_description.json")
    )
    with open(data_description_json_path[0], "r") as f:
        data_description_json = json.load(f)

    logger.info(
        f"Found primary data {data_description_json['name']}. \
        Starting acquisition nwb packaging now"
    )

    response = docdb_api_client.retrieve_docdb_records(
        filter_query={"data_description.name": data_description_json["name"]}
    )[0]

    write_ancillary_metadata(response, settings.output_directory)

    # pull version from here - file always assumed to exist at that path
    task_input_logic_path = (
        primary_data_path[0] / "behavior" / "Logs" / "tasklogic_input.json"
    )
    if not task_input_logic_path.exists():
        raise FileNotFoundError(
            f"No task logic input file found at path {task_input_logic_path}"
        )

    with open(task_input_logic_path, "r") as f:
        task_input_logic = json.load(f)

    try:
        contract_version = task_input_logic["version"]
    except KeyError:
        contract_version = task_input_logic["schema_version"]    

    if contract_version == "0.3.0":
        from v0_3_0 import dataset
    else:
        from aind_behavior_vr_foraging.data_contract import dataset
    
    logger.info(
        f"Running contract version: {contract_version}"
    )
    vr_foraging_dataset = dataset(
        primary_data_path[0], version=contract_version
    )

    # exec = vr_foraging_dataset["Behavior"].load_all()  # load tree structure
    # streams = tuple(vr_foraging_dataset.iter_all())
 
    processor = DatasetProcessor(vr_foraging_dataset, primary_data_path[0], raise_on_error=False)
    processed_sites = processor.process()


    nwb_file = create_base_nwb_file(primary_data_path[0])
    processing_module = nwb_file.create_processing_module(
        name="Behavior", description="Behavior Data - Velocity, Sniffing, and Licks"
    )
    
    filtered_velocity = processor.get_velocity()
    velocity_series = TimeSeries(
        name="Velocity",
        data=filtered_velocity["filtered_velocity"].values,
        unit="cm/s",
        timestamps=filtered_velocity.index.values,
    )
    processing_module.add(velocity_series)
    
    sniffing = processor.get_sniffing()
    sniffing_series = TimeSeries(
        name="Sniffing",
        data=sniffing.values,
        unit="V",
        timestamps=sniffing.index.values,
    )
    processing_module.add(sniffing_series)
    
    licks = processor.get_licks()
    lick_series = TimeSeries(
        name="Licks",
        data=licks.values,
        unit="V",
        timestamps=licks.index.values,
    )
    processing_module.add(lick_series)
    # for stream in streams:
    #     if stream.is_collection:  # only process leaf nodes into nwb
    #         continue

    #     name = stream.resolved_name.replace("::", ".")
    #     name = name[name.index(".") + 1:]
    #     if isinstance(stream, data_contract.harp.HarpRegister) or isinstance(
    #         stream, data_contract.csv.Csv
    #     ):
    #         try:
    #             dynamic_table = pynwb.core.DynamicTable.from_dataframe(
    #                 name=name,
    #                 table_description=stream.description,
    #                 df=stream.data.reset_index(),
    #             )
    #             nwb_file.add_acquisition(dynamic_table)
    #         except (ValueError, FileNotFoundError) as e:
    #             logger.error(
    #                 f"Failed to load {stream.name} with exception {e}"
    #             )
    #     elif isinstance(stream, data_contract.json.SoftwareEvents):
    #         try:
    #             data = utils.clean_dataframe_for_nwb(stream.data.reset_index())
    #             dynamic_table = pynwb.core.DynamicTable.from_dataframe(
    #                 name=name, table_description=stream.description, df=data
    #             )
    #             nwb_file.add_acquisition(dynamic_table)
    #         except (ValueError, FileNotFoundError) as e:
    #             logger.error(
    #                 f"Failed to get {stream.name} \
    #                 from {stream.parent} with error {e}"
    #             )
    #     elif isinstance(stream, data_contract.json.PydanticModel):
    #         try:
    #             data = utils.clean_dictionary_for_nwb(stream.data.model_dump())

    #             nwb_file.add_acquisition(
    #                 pynwb.core.DynamicTable(
    #                     name=name,
    #                     description=json.dumps(data),
    #                 )
    #             )
    #         except (ValidationError) as e:
    #             logger.error(
    #                 f"Failed to get {stream.name} \
    #                 from {stream.parent} with error {e}"
    #             )

    for field_name, field in Site.model_fields.items():
        if field_name in ["start_time", "stop_time"]:
            continue
        nwb_file.add_trial_column(name=field_name, description=field.description)

    for site in processed_sites:
        nwb_file.add_trial(**site.model_dump())

    nwb_result_path = (
        settings.output_directory / f"behavior.nwb.zarr"
    )
    logger.info(
        "Succesfully finished nwb packaging."
    )
    logger.info(f"Writing to disk now at path {nwb_result_path} as zarr")
    with NWBZarrIO(
        (nwb_result_path).as_posix(),
        "w",
    ) as io:
        io.write(nwb_file)
    logger.info(f"NWB zarr successfully written to path {nwb_result_path}")

    end_process_time = datetime.now()
    data_process = DataProcess(
        start_date_time=start_process_time,
        end_date_time=end_process_time,
        stage=ProcessStage.PROCESSING,
        process_type=ProcessName.PIPELINE,
        experimenters=["Arjun Sridhar"],
        code=Code(
            url=GITHUB_URL,
            version=VERSION
        ),
        output_parameters={},
        notes=f"Run with data contract version: {contract_version}"
    )
    with open(settings.output_directory / "data_process.json", "w") as f:
        f.write(data_process.model_dump_json(indent=4))
