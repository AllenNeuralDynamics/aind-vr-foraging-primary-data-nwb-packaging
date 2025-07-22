import json
import logging
from pathlib import Path

import contraqctor.contract as data_contract
import pynwb
from aind_behavior_vr_foraging.data_contract import dataset
from dateutil import parser
from hdmf_zarr import NWBZarrIO
from ndx_events import NdxEventsNWBFile
from pydantic import Field
from pydantic_settings import BaseSettings

import utils

logger = logging.getLogger(__name__)


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
    primary_data_path = tuple(settings.input_directory.glob("*"))
    if not primary_data_path:
        raise FileNotFoundError("No primary data asset attached")

    if len(primary_data_path) > 1:
        raise ValueError(
            "Multiple primary data assets attached. Only single asset needed"
        )

    session_json_path = tuple(settings.input_directory.glob("*/session.json"))
    data_description_json_path = tuple(
        settings.input_directory.glob("*/data_description.json")
    )
    if not session_json_path:
        raise FileNotFoundError("Primary data asset has no session json file")
    if not data_description_json_path:
        raise FileNotFoundError(
            "Primary data asset has no data description json"
        )

    with open(session_json_path[0], "r") as f:
        session_json = json.load(f)
    with open(data_description_json_path[0], "r") as f:
        data_description_json = json.load(f)
    logger.info(
        f"Found primary data {data_description_json['name']}. \
        Starting acquisition nwb packaging now"
    )

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

    contract_version = task_input_logic["version"]
    logger.info(f"Using data contract version {contract_version}")
    vr_foraging_dataset = dataset(
        primary_data_path[0], version=contract_version
    )
    exec = vr_foraging_dataset["Behavior"].load_all()  # load tree structure
    streams = tuple(vr_foraging_dataset.iter_all())
    event_data = []  # for adding to events table

    # using this ndx object for events table
    nwb_file = NdxEventsNWBFile(
        session_id=data_description_json["name"],
        session_description=session_json["session_type"],
        session_start_time=parser.parse(session_json["session_start_time"]),
        identifier=data_description_json["subject_id"],
    )
    for stream in streams:
        if stream.is_collection:  # only process leaf nodes into nwb
            continue

        name = stream.resolved_name.replace("::", ".")
        name = name[name.index(".") + 1:]
        if isinstance(
            stream, data_contract.harp.HarpRegister
        ) or isinstance(stream, data_contract.csv.Csv):
            try:
                dynamic_table = pynwb.core.DynamicTable.from_dataframe(
                    name=name,
                    table_description=stream.description,
                    df=stream.data.reset_index(),
                )
                nwb_file.add_acquisition(dynamic_table)
            except (ValueError, FileNotFoundError) as e:
                logger.error(
                    f"Failed to load {stream.name} with exception {e}"
                )
        elif isinstance(stream, data_contract.json.SoftwareEvents):
            try:
                data = utils.clean_dataframe_for_nwb(stream.data.reset_index())
                dynamic_table = pynwb.core.DynamicTable.from_dataframe(
                    name=name, table_description=stream.description, df=data
                )
                event_data.append(dynamic_table)
                nwb_file.add_acquisition(dynamic_table)
            except (ValueError, FileNotFoundError) as e:
                logger.error(
                    f"Failed to get {stream.name} \
                    from {stream.parent} with error {e}"
                )
        elif isinstance(stream, data_contract.json.PydanticModel):
            data = utils.clean_dictionary_for_nwb(stream.data.model_dump())

            nwb_file.add_acquisition(
                pynwb.core.DynamicTable(
                    name=name,
                    description=json.dumps(data),
                )
            )

    nwb_result_path = (
        settings.output_directory / f"{data_description_json['name']}_nwb"
    )
    logger.info(
        "Succesfully finished nwb acquisition packaging with timeseries."
    )
    logger.info(f"Writing to disk now at path {nwb_result_path} as zarr")
    with NWBZarrIO(
        (nwb_result_path).as_posix(),
        "w",
    ) as io:
        io.write(nwb_file)
    logger.info(f"NWB zarr successfully written to path {nwb_result_path}")
