"""Creates the event and meanings structure"""
from models import Site
import contraqctor
import numpy as np
import pandas as pd

from typing import Optional
from pydantic import BaseModel, Field
from processing import DatasetProcessor

class RewardSite(BaseModel):
    """Mouse enters and exits a virtual reward site."""

    name: str = Field(
        default="reward_site",
        description="Name of the event type."
    )
    start_time: float = Field(
        description="Start time of the event in software time. (unit: second)"
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event in software time. (unit: second)"
    )
    site_index: float = Field(
        description="Index identifier of the reward site."
    )


class Odor(BaseModel):
    """Onset and offset of the odor stimulus at a reward site."""

    name: str = Field(
        default="odor",
        description="Name of the event type."
    )
    start_time: float = Field(
        description="Start time of the event in software time. (unit: second)"
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event in software time. (unit: second)"
    )
    site_index: float = Field(
        description="Index identifier of the reward site."
    )
    concentration: list[float] = Field(
        description="Concentration levels of each odor channel. (unit: percent)"
    )


class Reward(BaseModel):
    """Water reward delivered to the mouse."""

    name: str = Field(
        default="reward",
        description="Name of the event type."
    )
    start_time: float = Field(
        description="Start time of the event in software time. (unit: second)"
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event in software time. (unit: second)"
    )
    site_index: float = Field(
        description="Index identifier of the reward site."
    )
    reward_volume: float = Field(
        description="Volume of water reward delivered. (unit: microliter)"
    )


class Lick(BaseModel):
    """Mouse licks the lick spout."""

    name: str = Field(
        default="lick",
        description="Name of the event type."
    )
    start_time: float = Field(
        description="Start time of the event in software time. (unit: second)"
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event in software time. (unit: second)"
    )


class Tone(BaseModel):
    """Auditory cue indicating mouse has fulfilled stop criteria."""

    name: str = Field(
        default="tone",
        description="Name of the event type."
    )
    start_time: float = Field(
        description="Start time of the event in software time. (unit: second)"
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event in software time. (unit: second)"
    )
    site_index: float = Field(
        description="Index identifier of the reward site."
    )


class Patch(BaseModel):
    """Mouse enters and exits a virtual patch."""

    name: str = Field(
        default="patch",
        description="Name of the event type."
    )
    start_time: float = Field(
        description="Start time of the event in software time. (unit: second)"
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event in software time. (unit: second)"
    )
    patch_index: str = Field(
        description="Label identifier of the patch."
    )


class Block(BaseModel):
    """Onset and offset of an experimental block."""

    name: str = Field(
        default="block",
        description="Name of the event type."
    )
    start_time: float = Field(
        description="Start time of the event in software time. (unit: second)"
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event in software time. (unit: second)"
    )
    block_index: float = Field(
        description="Index identifier of the block."
    )


class InterSite(BaseModel):
    """Mouse traverses an inter-site interval."""

    name: str = Field(
        default="inter_site",
        description="Name of the event type."
    )
    start_time: float = Field(
        description="Start time of the event in software time. (unit: second)"
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event in software time. (unit: second)"
    )
    site_index: float = Field(
        description="Index identifier of the adjacent reward site."
    )


class InterPatch(BaseModel):
    """Mouse traverses an inter-patch interval."""

    name: str = Field(
        default="inter_patch",
        description="Name of the event type."
    )
    start_time: float = Field(
        description="Start time of the event in software time. (unit: second)"
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event in software time. (unit: second)"
    )
    site_index: float = Field(
        description="Index identifier of the site"
    )


class RewardProbability(BaseModel):
    """Reward probability sampled by the task logic."""

    name: str = Field(
        default="reward_probability",
        description="Name of the event type."
    )
    start_time: float = Field(
        description="Start time of the event in software time. (unit: second)"
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event in software time. (unit: second)"
    )
    site_index: float = Field(
        description="Index identifier of the reward site."
    )


class Stop(BaseModel):
    """Mouse velocity crosses the stop threshold."""

    name: str = Field(
        default="stop",
        description="Name of the event type."
    )
    start_time: float = Field(
        description="Start time of the event in software time. (unit: second)"
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event in software time. (unit: second)"
    )


AnyEvent = (
    RewardSite
    | Odor
    | Reward
    | Lick
    | Tone
    | Patch
    | Block
    | InterSite
    | InterPatch
    | RewardProbability
    | Stop
)

def extract_patch_events(patches: pd.DataFrame) -> list[Patch]:
    """Extract patch onset and offset events from the patches dataframe.

    Parameters
    ----------
    patches : pd.DataFrame
        Dataframe of ActivePatch events with a 'data' column containing
        a 'label' field.

    Returns
    -------
    list[Patch]
        List of Patch events with start and stop times.
    """
    patch_events = []

    patches = patches.copy()
    patches["label"] = patches["data"].apply(lambda x: x["label"])

    # Detect label changes
    label_changes = patches["label"] != patches["label"].shift(1)
    patch_starts = patches[label_changes]

    for i, (timestamp, row) in enumerate(patch_starts.iterrows()):
        # stop time is the start of the next patch, or None if last
        if i + 1 < len(patch_starts):
            stop_time = patch_starts.index[i + 1]
        else:
            stop_time = None

        patch_events.append(Patch(
            start_time=timestamp,
            stop_time=stop_time,
            patch_index=row["data"]["label"],
        ))

    return patch_events

def extract_block_events(blocks: pd.DataFrame) -> list[Block]:
    """Extract block onset and offset events from the blocks dataframe.
    Each row represents a new block.

    Parameters
    ----------
    blocks : pd.DataFrame
        Dataframe of Block events where each row is a new block.

    Returns
    -------
    list[Block]
        List of Block events with start and stop times.
    """
    block_events = []

    for i, (timestamp, _) in enumerate(blocks.iterrows()):
        if i + 1 < len(blocks):
            stop_time = blocks.index[i + 1]
        else:
            stop_time = None

        block_events.append(Block(
            start_time=timestamp,
            stop_time=stop_time,
            block_index=i,
        ))

    return block_events

def extract_stop_events(
    velocity: pd.DataFrame,
    stop_threshold: pd.DataFrame,
) -> list[Stop]:
    """Extract stop onset and offset events by comparing absolute
    velocity against the stop velocity threshold.

    Parameters
    ----------
    velocity : pd.DataFrame
        Velocity dataframe with a 'filtered_velocity' column.
    stop_threshold : pd.DataFrame
        StopVelocityThreshold dataframe with a 'data' column
        containing the threshold value at each timestamp.

    Returns
    -------
    list[Stop]
        List of Stop events with start and stop times.
    """
    velocity = velocity.dropna(subset=["filtered_velocity"])

    # Align threshold to velocity timestamps
    threshold = pd.merge_asof(
        velocity.sort_index(),
        stop_threshold[["data"]].rename(
            columns={"data": "threshold"}
        ).sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )["threshold"]

    is_stopped = velocity["filtered_velocity"].abs() < threshold

    # Detect transitions
    transitions = is_stopped.astype(int).diff()
    stop_onsets = transitions[transitions == 1].index
    stop_offsets = transitions[transitions == -1].index

    events = []
    for onset in stop_onsets:
        subsequent = stop_offsets[stop_offsets > onset]
        offset = subsequent[0] if len(subsequent) > 0 else None
        events.append(Stop(
            start_time=onset,
            stop_time=offset,
        ))

    return events

def generate_event_list(
    processor: DatasetProcessor,
    sites: list[Site]
) -> list[AnyEvent]:
    """Create an event table from a dataset.

    Parameters
    ----------
    processor : DatasetProcessor
        The processor object for getting streams
    
    sites: list[Site]
        The site objects for pulling events from

    Returns
    -------
    list[AnyEvent]
        List of all events to be used to make event table
    """
    events = []
    dataset = processor.dataset

    for site in sites:
        if site.site_label == "RewardSite":
            events.append(
                RewardSite(
                    start_time=site.start_time,
                    stop_time=site.stop_time,
                    site_index=site.site_index
                )
            )
            # believe can only happen in reward site
            if not np.isnan(site.odor_onset_time):
                odor_times = dataset.at("Behavior").at("HarpOlfactometer").load().at("EndValveState").load().data
                idx = odor_times.index.get_loc(site.odor_onset_time)
                odor_event_offset_time = odor_times.index[idx + 1]
                events.append(
                    Odor(
                        start_time=site.odor_onset_time,
                        stop_time=odor_event_offset_time,
                        site_index=site.site_index,
                        concentration=site.odor_concentration
                    )
                )
        
        if not np.isnan(site.reward_onset_time):
            events.append(
                Reward(
                    start_time=site.reward_onset_time,
                    site_index=site.site_index,
                    reward_volume=site.reward_amount
                )
            )
        
        if not np.isnan(site.choice_cue_time):
            events.append(
                Tone(
                    start_time=site.choice_cue_time,
                    site_index=site.site_index
                )
            )
        
        if site.site_label == "InterSite":
            events.append(
                InterSite(
                    start_time=site.start_time,
                    stop_time=site.stop_time,
                    site_index=site.site_index
                )
            )

        if site.site_label == "InterPatch":
            events.append(
                InterPatch(
                    start_time=site.start_time,
                    stop_time=site.stop_time,
                    site_index=site.site_index
                )
            )
        
        if not np.isnan(site.reward_probability):
            events.append(
                RewardProbability(
                    start_time=site.start_time,
                    site_index=site.site_index
                )
            )
    
    lick_times = processor.get_licks().index
    for lick_time in lick_times:
        events.append(
            Lick(
                start_time=lick_time
            )
        )
    
    patches = dataset.at("Behavior").at("SoftwareEvents").at("ActivePatch").load().data
    patch_events = extract_patch_events(patches)
    for patch_event in patch_events:
        events.append(patch_event)

    blocks = dataset.at("Behavior").at("SoftwareEvents").at("Block").load().data
    block_events = extract_block_events(blocks)
    for block_event in block_events:
        events.append(block_event)
    
    velocity = processor.get_velocity()
    stop_threshold = dataset.at("Behavior").at("SoftwareEvents").at("StopVelocityThreshold").load().data
    stop_events = extract_stop_events(velocity, stop_threshold)
    for stop_event in stop_events:
        events.append(stop_event)

    return events

def events_to_dataframe(events: list[AnyEvent]) -> pd.DataFrame:
    """Convert a list of events to a dataframe.

    Parameters
    ----------
    events : list[AnyEvent]
        List of events from the VR foraging task.

    Returns
    -------
    pd.DataFrame
        Dataframe with one row per event onset/offset.
    """
    rows = []

    for event in events:
        base = {
            "site_index": getattr(event, "site_index", np.nan),
            "patch_index": getattr(event, "patch_index", np.nan),
            "block_index": getattr(event, "block_index", np.nan),
            "concentration": getattr(event, "concentration", np.nan),
            "reward_volume": getattr(event, "reward_volume", np.nan),
        }

        rows.append({
            "timestamp": event.start_time,
            "event_type": f"{event.name}_onset",
            **base,
        })

        if event.stop_time is not None:
            rows.append({
                "timestamp": event.stop_time,
                "event_type": f"{event.name}_offset",
                **base,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df