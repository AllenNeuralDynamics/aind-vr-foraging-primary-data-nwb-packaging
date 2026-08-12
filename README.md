# VR Foraging Primary Data NWB Packaging

The capsule can be found [here](https://codeocean.allenneuraldynamics.org/capsule/3265591/tree).

This capsule is a thin wrapper around the [`aind-behavior-vr-foraging-packaging`](https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.Packaging) library. All packaging logic — data-stream processors, NWB layout, and how to read the resulting file — lives in that package and is documented there. `run_capsule.py` calls the library, and writes the results.

**Changes to the packaging pipeline should be made as PRs to [`Aind.Behavior.VrForaging.Packaging`](https://github.com/AllenNeuralDynamics/Aind.Behavior.VrForaging.Packaging), not here.** This capsule should only change when the pinned library version is bumped or when the Code Ocean wiring itself (data contract resolution, output paths, environment) needs updating.

## Input

A single raw acquisition directory mounted at `/data`. Input data should follow the standard defined [here](https://github.com/AllenNeuralDynamics/aind-file-standards/blob/main/docs/core/core-standards.md).

```
/data/
└── <asset_name>/
    ├── acquisition.json
    ├── data_description.json
    ├── instrument.json
    ├── procedures.json
    ├── processing.json
    ├── subject.json
    └── behavior/
        ├── Behavior.harp/
        ├── SoftwareEvents/
        ├── Logs/
        └── ...
```

## Output

Files written to `/results`:

| File | Description |
| --- | --- |
| `behavior.nwb.zarr` | The packaged NWB file (Zarr backend), following the file standards listed [here](https://github.com/AllenNeuralDynamics/aind-file-standards/blob/main/docs/file_formats/nwb.md) |
| `data_process.json` | Provenance following the `aind-data-schema` `DataProcess` model, recording the packaging library and dataset versions (also mirrored on the file in `nwb.was_generated_by`) |
