from pathlib import Path

from aind_behavior_core_analysis import Dataset, DataStreamCollection
from aind_behavior_core_analysis.csv import Csv
from aind_behavior_core_analysis.harp import DeviceYmlByFile, HarpDevice
from aind_behavior_core_analysis.json import PydanticModel, SoftwareEvents
from aind_behavior_core_analysis.mux import MapFromPaths
from aind_behavior_core_analysis.text import Text
from aind_behavior_services.rig import AindBehaviorRigModel
from aind_behavior_services.session import AindBehaviorSessionModel
from aind_behavior_services.task_logic import AindBehaviorTaskLogicModel


def get_data_contract(dataset_root: Path) -> Dataset:
    """
    Load and return the data contract from the specified root path.

    Parameters
    ----------
    dataset_root : Path
        The root directory where the dataset is stored. This should point to the
        top-level folder containing the necessary data files.

    Returns
    -------
    Dataset
        A `Dataset` object which models the data contract.
    """
    my_dataset = Dataset(
        name="my_dataset",
        version="1.0",
        description="My dataset",
        data_streams=DataStreamCollection(
            name="Dataset",
            description="Root of the dataset",
            data_streams=[
                DataStreamCollection(
                    name="Behavior",
                    description="Data from the Behavior modality",
                    data_streams=[
                        HarpDevice(
                            name="HarpBehavior",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/Behavior.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpManipulator",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/StepperDriver.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpTreadmill",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/Treadmill.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpOlfactometer",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/Olfactometer.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpSniffDetector",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/SniffDetector.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpLickometer",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/Lickometer.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpClockGenerator",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/ClockGenerator.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        HarpDevice(
                            name="HarpEnvironmentSensor",
                            reader_params=HarpDevice.make_params(
                                path=dataset_root / "behavior/EnvironmentSensor.harp",
                                device_yml_hint=DeviceYmlByFile(),
                            ),
                        ),
                        DataStreamCollection(
                            name="HarpCommands",
                            description="Commands sent to Harp devices",
                            data_streams=[
                                HarpDevice(
                                    name="HarpBehavior",
                                    reader_params=HarpDevice.make_params(
                                        path=dataset_root
                                        / "behavior/HarpCommands/Behavior.harp",
                                        device_yml_hint=DeviceYmlByFile(),
                                    ),
                                ),
                                HarpDevice(
                                    name="HarpManipulator",
                                    reader_params=HarpDevice.make_params(
                                        path=dataset_root
                                        / "behavior/HarpCommands/StepperDriver.harp",
                                        device_yml_hint=DeviceYmlByFile(),
                                    ),
                                ),
                                HarpDevice(
                                    name="HarpTreadmill",
                                    reader_params=HarpDevice.make_params(
                                        path=dataset_root
                                        / "behavior/HarpCommands/Treadmill.harp",
                                        device_yml_hint=DeviceYmlByFile(),
                                    ),
                                ),
                                HarpDevice(
                                    name="HarpOlfactometer",
                                    reader_params=HarpDevice.make_params(
                                        path=dataset_root
                                        / "behavior/HarpCommands/Olfactometer.harp",
                                        device_yml_hint=DeviceYmlByFile(),
                                    ),
                                ),
                                HarpDevice(
                                    name="HarpSniffDetector",
                                    reader_params=HarpDevice.make_params(
                                        path=dataset_root
                                        / "behavior/HarpCommands/SniffDetector.harp",
                                        device_yml_hint=DeviceYmlByFile(),
                                    ),
                                ),
                                HarpDevice(
                                    name="HarpLickometer",
                                    reader_params=HarpDevice.make_params(
                                        path=dataset_root
                                        / "behavior/HarpCommands/Lickometer.harp",
                                        device_yml_hint=DeviceYmlByFile(),
                                    ),
                                ),
                                HarpDevice(
                                    name="HarpClockGenerator",
                                    reader_params=HarpDevice.make_params(
                                        path=dataset_root
                                        / "behavior/HarpCommands/ClockGenerator.harp",
                                        device_yml_hint=DeviceYmlByFile(),
                                    ),
                                ),
                                HarpDevice(
                                    name="HarpEnvironmentSensor",
                                    reader_params=HarpDevice.make_params(
                                        path=dataset_root
                                        / "behavior/HarpCommands/EnvironmentSensor.harp",
                                        device_yml_hint=DeviceYmlByFile(),
                                    ),
                                ),
                            ],
                        ),
                        MapFromPaths(
                            name="SoftwareEvents",
                            description="Software events generated by the workflow. The timestamps of these events are low precision and should not be used to align to physiology data.",
                            reader_params=MapFromPaths.make_params(
                                paths=[
                                    dataset_root / "behavior/SoftwareEvents",
                                    dataset_root / "behavior/UpdaterEvents",
                                ],
                                include_glob_pattern=["*.json"],
                                inner_data_stream=SoftwareEvents,
                                inner_param_factory=lambda p: SoftwareEvents.make_params(
                                    path=p,
                                    index="timestamp",
                                ),
                            ),
                        ),
                        MapFromPaths(
                            name="OperationControl",
                            description="Events related with conditions and task logic computed online.",
                            reader_params=MapFromPaths.make_params(
                                paths=dataset_root / "behavior/OperationControl",
                                include_glob_pattern=["*.csv"],
                                inner_data_stream=Csv,
                                inner_param_factory=lambda p: Csv.make_params(
                                    path=p,
                                    index="Seconds",
                                ),
                            ),
                        ),
                        Csv(
                            name="RendererSynchState",
                            description="Contains information that allows the post-hoc alignment of visual stimuli to the behavior data.",
                            reader_params=Csv.make_params(
                                path=dataset_root
                                / "behavior/Renderer/RendererSynchState.csv"
                            ),
                        ),
                        DataStreamCollection(
                            name="Logs",
                            data_streams=[
                                Text(
                                    name="Launcher",
                                    description="Contains the console log of the launcher process.",
                                    reader_params=Text.make_params(
                                        path=dataset_root
                                        / "behavior/Logs/launcher.log",
                                    ),
                                ),
                                SoftwareEvents(
                                    name="EndSession",
                                    description="A file that determines the end of the session. If the file is empty, the session is still running or it was not closed properly.",
                                    reader_params=SoftwareEvents.make_params(
                                        path=dataset_root
                                        / "behavior/Logs/endsession.json",
                                    ),
                                ),
                            ],
                        ),
                        DataStreamCollection(
                            name="InputSchemas",
                            description="Configuration files for the behavior rig, task_logic and session.",
                            data_streams=[
                                PydanticModel(
                                    name="Rig",
                                    reader_params=PydanticModel.make_params(
                                        model=AindBehaviorRigModel,
                                        path=dataset_root
                                        / "behavior/Logs/rig_input.json",
                                    ),
                                ),
                                PydanticModel(
                                    name="TaskLogic",
                                    reader_params=PydanticModel.make_params(
                                        model=AindBehaviorTaskLogicModel,
                                        path=dataset_root
                                        / "behavior/Logs/tasklogic_input.json",
                                    ),
                                ),
                                PydanticModel(
                                    name="Session",
                                    reader_params=PydanticModel.make_params(
                                        model=AindBehaviorSessionModel,
                                        path=dataset_root
                                        / "behavior/Logs/session_input.json",
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )

    return my_dataset
