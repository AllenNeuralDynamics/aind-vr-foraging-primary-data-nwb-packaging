import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_dataframe_for_nwb(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a pandas DataFrame to ensure compatibility with NWB format.

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


def datetime_to_str_in_dict(
    data: Union[Dict[str, Any], List[Any], datetime, Any],
) -> Union[Dict[str, Any], List[Any], str, Any]:
    """
    Recursively convert all `datetime` objects in a
    nested dictionary or list to strings.

    Parameters
    ----------
    data : dict or list or datetime or any
        Input data structure which may contain
        nested dictionaries, lists, and datetime objects.

    Returns
    -------
    dict or list or str or any
    All values in nested dict converted from datetime to string
    """
    if isinstance(data, dict):
        return {k: datetime_to_str_in_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [datetime_to_str_in_dict(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()
    else:
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

    return datetime_to_str_in_dict(data)
