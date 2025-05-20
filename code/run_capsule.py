import json
import logging
from datetime import datetime

import data_contract
import pandas as pd
import pynwb
import utils
from aind_behavior_core_analysis._core import DataStream
from aind_behavior_core_analysis.utils import load_branch
from dateutil import parser
from hdmf_zarr import NWBZarrIO
from ndx_events import EventsTable, MeaningsTable, NdxEventsNWBFile

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    primary_data_path = tuple(utils.DATA_PATH.glob("*"))
    if not primary_data_path:
        raise FileNotFoundError("No primary data asset attached")

    if len(primary_data_path) > 1:
        raise ValueError(
            "Multiple primary data assets attached. Only single asset needed"
        )

    session_json_path = tuple(utils.DATA_PATH.glob("*/session.json"))
    data_description_json_path = tuple(utils.DATA_PATH.glob("*/data_description.json"))
    if not session_json_path:
        raise FileNotFoundError("Primary data asset has no session json file")
    if not data_description_json_path:
        raise FileNotFoundError("Primary data asset has no data description json")

    with open(session_json_path[0], "r") as f:
        session_json = json.load(f)
    with open(data_description_json_path[0], "r") as f:
        data_description_json = json.load(f)
    logger.info(
        f"Found primary data {data_description_json["name"]}. Starting acquisition nwb packaging now"
    )

    dataset = data_contract.get_data_contract(primary_data_path[0])
    exec = load_branch(dataset.data_streams)  # load tree structure
    streams = tuple(dataset.data_streams.walk_data_streams())  # get all data streams
    top_level_stream = utils.get_top_level_stream(streams)
    event_data = []  # for adding to events table

    # using this ndx object for events table
    nwb_file = NdxEventsNWBFile(
        session_id=data_description_json["name"],
        session_description=session_json["session_type"],
        session_start_time=parser.parse(session_json["session_start_time"]),
        identifier=data_description_json["subject_id"],
    )
    for stream in streams:
        if (
            stream.is_collection
        ):  # only process leaf nodes which is what will be ultimately packaged into nwb
            continue

        name = utils.get_stream_name(stream, top_level_stream)
        if isinstance(stream.parent, data_contract.HarpDevice) or isinstance(
            stream, data_contract.Csv
        ):
            try:
                dynamic_table = pynwb.core.DyanmicTable.from_dataframe(
                    name=name,
                    table_description=stream.description,
                    df=stream.data.reset_index(),
                )
                nwb_file.add_acquisition(dynamic_table)
            except (ValueError, FileNotFoundError) as e:
                logger.info(f"Failed to load {stream.name} with exception {e}")
        elif isinstance(stream, data_contract.SoftwareEvents):
            try:
                data = utils.clean_dataframe_for_nwb(stream.data.reset_index())
                dynamic_table = pynwb.core.DynamicTable.from_dataframe(
                    name=name, table_description=stream.description, df=data
                )
                event_data.append(dynamic_table)
                nwb_file.add_acquisition(dynamic_table)
            except (ValueError, FileNotFoundError) as e:
                logger.info(
                    f"Failed to get {stream.name} from {stream.parent} with error {e}"
                )
        elif isinstance(stream, data_contract.PydanticModel):
            data = utils.clean_dictionary_for_nwb(stream.data.model_dump())

            nwb_file.add_acquisition(
                pynwb.core.DynamicTable(
                    name=name,
                    description=json.dumps(data),
                )
            )

    meanings_table = MeaningsTable(
        name="event_descriptions",
        description="Describes meaning of event and the corresponding data",
    )
    events_table = EventsTable(
        name="events",
        description="Events logged by acquisition workflow",
        meanings_tables=[meanings_table],
    )
    events_table.add_column(
        name="event_name", description="event logged by acquisition workflow"
    )
    events_table.add_column(
        name="event_data", description="event data from the acquisition workflow"
    )

    for table in event_data:
        utils.add_event(events_table, meanings_table, table)

    nwb_file.add_events_table(events_table)

    nwb_result_path = (
        utils.RESULTS_PATH / f"{data_description_json['name']}_primary_nwb"
    )
    logger.info(
        f"Succesfully finished nwb acquisition packaging with timeseries and events. Writing to disk now at path {nwb_result_path} as zarr"
    )
    with NWBZarrIO(
        (nwb_result_path).as_posix(),
        "w",
    ) as io:
        io.write(nwb_file)
    logger.info(f"NWB zarr successfully written to path {nwb_result_path}")
