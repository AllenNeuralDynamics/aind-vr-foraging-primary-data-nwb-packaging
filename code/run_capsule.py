import json
import logging
from datetime import datetime

import data_contract
import pandas as pd
import pynwb
import utils
from aind_behavior_core_analysis.utils import load_branch
from dateutil import parser
from hdmf_zarr import NWBZarrIO
from ndx_events import EventsTable, MeaningsTable, NdxEventsNWBFile

logger = logging.getLogger(__name__)


def add_event(
    events_table: EventsTable,
    meanings_table: MeaningsTable,
    event_data: pynwb.core.DynamicTable,
) -> None:
    """
    Adds event data to the EventsTable and MeaningsTable based on the provided DataFrame.

    Parameters
    ----------
    events_table : EventsTable
        A table where event data is stored. This table will be updated with new event information.

    meanings_table : MeaningsTable
        A table containing the meanings of events. This table will be updated based on the data provided.

    data : pynwb.core.DynamicTable
        A table containing new event data. Each row corresponds to a new event entry and contains
        information that needs to be inserted into both the events_table and meanings_table.

    Returns
    -------
    None
    """
    data = event_data[:]
    for index, row in data.iterrows():
        events_table.add_row(
            timestamp=row["timestamp"], event_name=row["name"], event_data=row["data"]
        )
        meanings_table.add_row(
            value=row["data"],
            meaning=f"{row['name']} - {event_data.description}",
        )


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

    dataset = data_contract.get_data_contract(primary_data_path[0])
    exec = load_branch(dataset.data_streams)  # load tree structure
    streams = tuple(dataset.data_streams.walk_data_streams())  # get all data streams
    timeseries_groups = {}
    event_groups = {}

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

        if isinstance(stream.parent, data_contract.HarpDevice):
            timeseries_groups = utils.get_harp_nwb_streams(timeseries_groups, stream)
        elif isinstance(stream, data_contract.SoftwareEvents):
            event_groups = utils.get_software_events_nwb_streams(event_groups, stream)
        elif isinstance(stream, data_contract.Csv):
            if stream.parent.name == "Behavior":  # add directly, no nested streams
                nwb_file.add_acquisition(
                    pynwb.core.DynamicTable.from_dataframe(
                        name=stream.name,
                        table_description=stream.description,
                        df=stream.data.reset_index(),
                    )
                )
            else:
                key = stream.parent.name
                timeseries_groups = utils.add_table_to_group(
                    timeseries_groups,
                    stream.data.reset_index(),
                    key,
                    stream.name,
                    stream.parent.description,
                )
        elif isinstance(stream, data_contract.PydanticModel):
            data = stream.data.model_dump()
            """
            if "name" in data:  # conflicting key with nwb, have to remove
                del data["name"]

            #custom_data = utils.custom_data_interface(name=stream.name, data=data)
            #custom_data.add_data()
            """
            for key in data:
                if isinstance(data[key], datetime):
                    data[key] = data[key].isoformat()
            nwb_file.add_acquisition(
                pynwb.core.DynamicTable(name=stream.name, description=json.dumps(data))
            )

    for group, table_group in timeseries_groups.items():
        for key, table in table_group.data.items():
            nwb_file.add_acquisition(table)

    for group, table_group in event_groups.items():
        for key, table in table_group.data.items():
            nwb_file.add_acquisition(table)

    meanings_table = MeaningsTable(
        name="event_descriptions", description="Describes event meaning"
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

    for group in event_groups:
        events_data = event_groups[group].data
        for event in events_data:
            add_event(events_table, meanings_table, events_data[event])

    nwb_file.add_events_table(events_table)

    with NWBZarrIO(
        (
            utils.RESULTS_PATH / f"{data_description_json['name']}_primary_nwb"
        ).as_posix(),
        "w",
    ) as io:
        io.write(nwb_file)
