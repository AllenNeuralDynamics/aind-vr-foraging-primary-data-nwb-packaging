import json
import logging
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
        data['description'] = self.description
        # Serialize grouped tables
        data['grouped_tables'] = {name: table.to_dict() for name, table in self.data.items()}
        return data

    @classmethod
    def from_dict(cls, data):
        """Override from_dict to deserialize the GroupedDynamicTable and its tables."""
        instance = super().from_dict(data)
        instance.description = data['description']
        instance.grouped_tables = {
            name: DynamicTable.from_dict(table_data) for name, table_data in data['grouped_tables'].items()
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
    table_name: str,
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
    table_name : str
        The name to assign to the new table within the `table_group`.
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
            name=f"{key}.{table_name}", table_description=description, df=data
        )
    )

    return group


def get_harp_nwb_streams(
    timeseries_groups: dict[str, table_group], stream: DataStream
) -> dict[str, table_group]:
    """
    Extract and add harp NWB time series data from a stream to existing table groups.

    Parameters
    ----------
    timeseries_groups : dict[str, table_group]
        A dictionary where each key maps to a `table_group` object containing time series tables.
    stream : DataStream
        The data stream object containing harp NWB time series data to be extracted.

    Returns
    -------
    dict[str, table_group]
        The updated dictionary of `table_group` objects with additional tables added from the stream.
    """
    if stream.parent.parent.name == "HarpCommands":
        key = stream.parent.parent.name
        description = (
            f"{key}: {stream.parent.name}.{stream.name} - {stream.description}"
        )
        table_name = f"{stream.parent.name}.{stream.name}"
        try:
            timeseries_groups = add_table_to_group(
                timeseries_groups,
                stream.data.reset_index(),
                key,
                table_name,
                description,
            )
        except (ValueError, FileNotFoundError) as e:
            logger.info(
                f"Failed to get {stream.name} from {stream.parent} - {stream.parent.parent.name} with error {e}"
            )
    else:
        key = stream.parent.name
        try:
            timeseries_groups = add_table_to_group(
                timeseries_groups,
                stream.data.reset_index(),
                key,
                stream.name,
                stream.description,
            )
        except ValueError as e:
            logger.info(f"Failed to get {stream.name} from {stream.parent} with error {e}")

    return timeseries_groups


def get_software_events_nwb_streams(
    event_groups: dict[str, table_group], stream: DataStream
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

    Returns
    -------
    dict[str, table_group]
        The updated dictionary of `table_group` objects, now containing the software events
        extracted from the stream.
    """
    key = stream.parent.name

    if stream.description is None:
        description = stream.parent.description
    else:
        description = stream.description
    try:
        data = stream.data.reset_index()
        for column in data.columns:
            data[column].replace({None: np.nan}, inplace=True)

        data['timestamp_source'] = data['timestamp_source'].apply(lambda x: x.value)
        data['data_type'] = data['data_type'].apply(lambda x: x.value)
        # Convert dicts to JSON strings
        data['data'] = data['data'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
        event_groups = add_table_to_group(
            event_groups, data, key, stream.name, description
        )
    except (ValueError, FileNotFoundError) as e:
        logger.info(f"Failed to get {stream.name} from {stream.parent} with error {e}")

    return event_groups
