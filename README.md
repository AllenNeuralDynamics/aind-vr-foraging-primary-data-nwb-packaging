# VR Foraging Primary Data NWB Packaging

This capsule packages VR Foraging Primary Data into NWB following the file standards listed here: https://github.com/AllenNeuralDynamics/aind-file-standards/blob/main/file_formats/nwb.md. The input data contract can be found at the following. TODO: INSERT LINK TO VR FORAGING TASK AND CONTRACT.

The output of this packaging is an nwb file with data in the `acquisition` and `events` modules. Most, if not all of the data in the nwb are stored as `DynamicTables`, a table representation of the various streams from the input data contract. To read the nwb and access the data, the following code snippet can be run - dependencies are `hdmf-zarr`, and `pynwb`.

```
from hdmf_zarr import NWBZarrIO
import json

# REPLACE WITH PATH TO NWB
with NWBZarrIO('path/to/nwb', 'r') as io:
  nwb = io.read()

keys = list(nwb.acquisition.keys())
# access one of streams from acquisition
data = nwb.acquisition[keys[0]]
print(data)
data_df = data[:] # gives a dataframe representation of table

# For fetching json files from the data contract, for now, they are stored in a dynamic table but in the description field. The way to recover it from the nwb is as follows:
json_dict = json.dumps(nwb.acquisition[key].description)
```

The acqusition module in the nwb is structured as follows:
## 📁 NWBFile: Acqusition Module

This module contains tables from the primary data. Each entry is a `DynamicTable` that can be accessed with the code snippet above.

### 📑 DynamicTables

- `Behavior.HarpBehavior.AnalogData`
- `Behavior.HarpBehavior.AssemblyVersion`
- `Behavior.HarpBehavior.Camera0Frame`
- `Behavior.HarpBehavior.Camera0Frequency`
- `Behavior.HarpBehavior.Camera1Frame`
- `Behavior.HarpBehavior.Camera1Frequency`
- `Behavior.HarpBehavior.ClockConfiguration`
- `Behavior.HarpBehavior.CoreVersionHigh`
- `...`
