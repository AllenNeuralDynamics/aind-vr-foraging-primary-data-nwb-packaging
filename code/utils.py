import json
import logging
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
from aind_behavior_core_analysis._core import DataStream
from pynwb.file import DynamicTable, NWBDataInterface

DATA_PATH = Path("/data")
RESULTS_PATH = Path("/results")

logger = logging.getLogger(__name__)


class table_group(NWBDataInterface):
    """Custom data interface to group multiple tables."""

    def __init__(self, name, description=""):
        super().__init__(name=name)
        self.data = {}

    def add_table(self, table: DynamicTable):
        """Add a table to the group."""
        if isinstance(table, DynamicTable):
            self.data[table.name] = table
        else:
            raise ValueError("The object must be an instance of DynamicTable.")

    def get_table(self, table_name):
        """Get a table by name."""
        return self.data.get(table_name)

    def to_dict(self):
        """Override to_dict to serialize the GroupedDynamicTable and its tables."""
        data = super().to_dict()  # Serialize the base data first
        data["description"] = self.description
        # Serialize grouped tables
        data["grouped_tables"] = {
            name: table.to_dict() for name, table in self.data.items()
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Override from_dict to deserialize the GroupedDynamicTable and its tables."""
        instance = super().from_dict(data)
        instance.description = data["description"]
        instance.grouped_tables = {
            name: DynamicTable.from_dict(table_data)
            for name, table_data in data["grouped_tables"].items()
        }
        return instance


class custom_data_interface(NWBDataInterface):
    """A custom data interface that can store a dictionary as attributes."""

    def __init__(self, name, data):
        super().__init__(name=name)
        self.data = data

    def add_data(self):
        """Add data as attributes."""
        for key, value in self.data.items():
            setattr(self, key, value)

    def get_data(self):
        """Return data as a dictionary of attribute names and values."""
        return {key: getattr(self, key) for key in self.data.keys()}


def add_table_to_group(
    group: dict[str, table_group],
    data: pd.DataFrame,
    key: str,
    description: str,
) -> dict[str, table_group]:
    """
    Add a DataFrame as a named table to a table group within a dictionary.

    Parameters
    ----------
    group : dict[str, table_group]
        A dictionary mapping string keys to `table_group` objects. Each `table_group`
        contains one or more named tables.
    data : pd.DataFrame
        The DataFrame to be added to the group.
    key : str
        The key identifying which `table_group` in the dictionary the table should be added to.
    description : str
        A description of what the table represents.

    Returns
    -------
    dict[str, table_group]
        The updated dictionary with the new table added to the specified group.
    """
    if key not in group:
        group[key] = table_group(name=key)

    group[key].add_table(
        pynwb.core.DynamicTable.from_dataframe(
            name=f"{key}", table_description=description, df=data
        )
    )

    return group


def get_stream_name(stream: DataStream, top_level_stream: DataStream) -> str:
    """
    Generates a name for a given data stream relative to a top-level stream.

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


def get_harp_nwb_streams(
    timeseries_groups: dict[str, table_group],
    stream: DataStream,
    top_level_stream: DataStream,
) -> dict[str, table_group]:
    """
    Extract and add harp NWB time series data from a stream to existing table groups.

    Parameters
    ----------
    timeseries_groups : dict[str, table_group]
        A dictionary where each key maps to a `table_group` object containing time series tables.
    stream : DataStream
        The data stream object containing harp NWB time series data to be extracted.
    top_level_stream: DataStream
        The data stream object at the desired top level. Used for naming in nwb table
    Returns
    -------
    dict[str, table_group]
        The updated dictionary of `table_group` objects with additional tables added from the stream.
    """

    name = get_stream_name(stream, top_level_stream)
    try:
        timeseries_groups = add_table_to_group(
            timeseries_groups,
            stream.data.reset_index(),
            name,
            stream.description,
        )
    except ValueError as e:
        logger.info(f"Failed to get {stream.name} from {stream.parent} with error {e}")

    return timeseries_groups


def get_software_events_nwb_streams(
    event_groups: dict[str, table_group],
    stream: DataStream,
    top_level_stream: DataStream,
):
    """
    Extracts and add software event data from an NWB stream to existing event groups.

    Parameters
    ----------
    event_groups : dict[str, table_group]
        A dictionary where each key maps to a `table_group` object that contains event tables.
    stream : DataStream
        The data stream object containing software event data in NWB format to be extracted and added
        to the event groups.
    top_level_stream: DataStream
        The data stream object at the desired top level. Used for naming in nwb table
    Returns
    -------
    dict[str, table_group]
        The updated dictionary of `table_group` objects, now containing the software events
        extracted from the stream.
    """
    name = get_stream_name(stream, top_level_stream)

    try:
        data = stream.data.reset_index()
        for column in data.columns:
            # convert to nwb allowable types
            data[column].replace({None: np.nan}, inplace=True)
            data[column] = data[column].apply(
                lambda x: x.value if isinstance(x, Enum) else x
            )
            data[column] = data[column].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )

        event_groups = add_table_to_group(event_groups, data, name, stream.description)
    except (ValueError, FileNotFoundError) as e:
        logger.info(f"Failed to get {stream.name} from {stream.parent} with error {e}")

    return event_groups
