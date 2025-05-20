import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
from aind_behavior_core_analysis._core import DataStream
from ndx_events import EventsTable, MeaningsTable

DATA_PATH = Path("/data")
RESULTS_PATH = Path("/results")

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

    event_data : pynwb.core.DynamicTable
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


def get_top_level_stream(
    streams: tuple[DataStream], key_to_match: str = "Behavior"
) -> DataStream:
    """
    Identifies and returns the top-level data stream from a tuple of streams.

    This function assumes there is one such stream that serves as the entry point or parent in the stream hierarchy.

    Parameters
    ----------
    streams : tuple of DataStream
        A tuple containing multiple DataStream objects from which the top-level stream
        will be identified.

    key_to_match: str
        The name to match the top-level stream that is desired

    Returns
    -------
    DataStream
        The top-level stream among the input streams.
    """
    top_level_stream = None
    for stream in streams:
        if (
            stream.parent.parent is None and stream.name == key_to_match
        ):  # found the desired top level stream
            top_level_stream = stream
            break

    return top_level_stream


def get_stream_name(stream: DataStream, top_level_stream: DataStream) -> str:
    """
    Generates a name for a given data stream relative to a top-level stream. Mainly for naming in the nwb tables.

    This function determines a name that uniquely identifies the `stream` within the context
    of the `top_level_stream`.

    Parameters
    ----------
    stream : DataStream
        The target data stream for which to generate a name.

    top_level_stream : DataStream
        The root or top-level data stream that provides context for the name generation.

    Returns
    -------
    str
        The name of the stream with respect to the top-level stream passed in
    """
    name = None

    while stream.name != top_level_stream.name:
        if name is None:
            name = stream.name
        else:
            name = f"{stream.name}.{name}"

        stream = stream.parent

    name = f"{top_level_stream.name}.{name}"
    return name


def clean_dataframe_for_nwb(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a pandas DataFrame to ensure compatibility with NWB format.

    This function performs:
    - Converting unsupported data types (e.g., object or mixed types) to strings or appropriate formats.
    - Replacing NoneType with NaNs

    Parameters
    ----------
    data : pd.DataFrame
        The cleaned input DataFrame for NWB compatibility

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame that adheres to NWB data types
    """
    for column in data.columns:
        # convert to nwb allowable types
        data[column].replace({None: np.nan}, inplace=True)
        data[column] = data[column].apply(
            lambda x: x.value if isinstance(x, Enum) else x
        )
        data[column] = data[column].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else x
        )

    return data


def clean_dictionary_for_nwb(data: dict) -> dict:
    """
    Clean a dictionary to ensure compatibility with the NWB format.
    Parameters
    ----------
    data : dict
        The input dictionary to be cleaned for NWB compatibility.

    Returns
    -------
    dict
        A cleaned dictionary with NWB-compliant data
    """
    for key in data:
        if isinstance(data[key], datetime):
            data[key] = data[key].isoformat()

    return data
