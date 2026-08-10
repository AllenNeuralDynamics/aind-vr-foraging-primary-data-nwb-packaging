# VR Foraging Primary Data NWB Packaging

The capsule can be found [here](https://codeocean.allenneuraldynamics.org/capsule/3265591/tree). 

This capsule packages VR Foraging Primary Data into NWB following the file standards listed [here](https://github.com/AllenNeuralDynamics/aind-file-standards/blob/main/file_formats/nwb.md). 

The capsule uses a data contract with relevants paths to raw data to be packaged. Details on the data contract and an example can be found at the [following](https://allenneuraldynamics.github.io/Aind.Behavior.VrForaging/dataset/). 

Packaging logic is provided by the [`aind-behavior-vr-foraging-packaging`](https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.Packaging) library. `run_capsule.py` uses `NwbSession` and `create_processors` from that library to build the NWB file, keeping the capsule thin and delegating all data-stream processing to the versioned package.

## Outputs

The capsule writes two files to the output directory:

- `behavior.nwb.zarr` — the packaged NWB file (Zarr format)
- `data_process.json` — provenance metadata following the `aind-data-schema` `DataProcess` model, recording the packaging library version and dataset version used

Each processor writes to a different part of the NWB file. Data is **not** all in `acquisition` — where you look depends on the stream:

| Data | Location in NWB | Written by |
| --- | --- | --- |
| Trials (sites) table | `nwb.trials` | `SiteTableProcessor` |
| Position | `nwb.processing["behavior"]["Position"]["position"]` | `PositionAndVelocityProcessor` |
| Velocity | `nwb.processing["behavior"]["velocity"]` | `PositionAndVelocityProcessor` |
| Licks | `nwb.processing["behavior"]["licks"]` | `LicksProcessor` |
| Sniffing | `nwb.processing["behavior"]["sniffing"]` | `SniffingProcessor` |
| Derived events | `nwb.events_tables["events"]` | `EventsProcessor` |
| Software events | `nwb.acquisition[...]` (one `DynamicTable` per event type) | `SoftwareEventsProcessor` |

> **Note:** datasets with schema version < 0.6.0 automatically use legacy processor variants (`LegacySiteTableProcessor`, `LegacyPositionAndVelocityProcessor`). The output locations are the same.

To read the NWB and access the data, the following code snippet can be run — dependencies are `hdmf-zarr` and `pynwb`:

```python
import json

from hdmf_zarr import NWBZarrIO

# REPLACE WITH PATH TO NWB
with NWBZarrIO('path/to/behavior.nwb.zarr', 'r') as io:
    nwb = io.read()

    # Trials (sites) table, as a dataframe
    trials_df = nwb.trials[:]

    # Continuous streams live in the "behavior" processing module
    behavior = nwb.processing["behavior"]
    position = behavior["Position"]["position"]  # SpatialSeries, cm
    velocity = behavior["velocity"]              # TimeSeries, cm/s
    licks = behavior["licks"]                    # TimeSeries, lick onset/offset
    sniffing = behavior["sniffing"]              # TimeSeries, V

    # Each has .data and .timestamps
    position_values = position.data[:]
    position_times = position.timestamps[:]

    # Derived events table
    events_df = nwb.events_tables["events"][:]
    # the `data` column is JSON-serialized
    payloads = [json.loads(d) for d in events_df["data"]]

    # Software events: one DynamicTable per event type in acquisition
    software_event_names = list(nwb.acquisition.keys())
    software_events_df = nwb.acquisition[software_event_names[0]][:]
```

Provenance (packaging library version and dataset version) is stored on the file in
`nwb.was_generated_by`, and mirrored in `data_process.json`.
