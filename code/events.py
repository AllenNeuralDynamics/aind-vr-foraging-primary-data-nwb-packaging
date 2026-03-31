"""Creates the event and meanings structure"""
from models import Site
import contraqctor
import numpy as np
import pandas as pd

from typing import Optional
from pydantic import BaseModel, Field
from processing import DatasetProcessor

# TODO: try to build these models from the sidecar json
# TODO: move this to package
class RewardSite(BaseModel):
    """Mouse enters and exits a virtual reward site."""

    name_onset: str = Field(default="reward_site_enter")
    name_offset: str = Field(default="reward_site_exit")
    start_time: float = Field(
        description="Start time of the event "
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event "
    )
    site_index: float = Field(
        description="Index identifier of the reward site."
    )


class Odor(BaseModel):
    """Onset and offset of the odor stimulus at a reward site."""

    name_onset: str = Field(default="odor_onset")
    name_offset: str = Field(default="odor_offset")
    start_time: float = Field(
        description="Start time of the event "
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event "
    )
    site_index: float = Field(
        description="Index identifier of the reward site."
    )
    concentration: list[float] = Field(
        description="Concentration levels of each odor channel. (unit: percent)"
    )


class Reward(BaseModel):
    """Water reward delivered to the mouse."""

    name_onset: str = Field(default="reward")
    name_offset: str = Field(default="reward")
    start_time: float = Field(
        description="Start time of the event "
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event "
    )
    site_index: float = Field(
        description="Index identifier of the reward site."
    )
    reward_volume: float = Field(
        description="Volume of water reward delivered. (unit: microliter)"
    )


class Lick(BaseModel):
    """Mouse licks the lick spout."""

    name_onset: str = Field(default="lick")
    name_offset: str = Field(default="lick")
    start_time: float = Field(
        description="Start time of the event "
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event "
    )


class Tone(BaseModel):
    """Auditory cue indicating mouse has fulfilled stop criteria."""

    name_onset: str = Field(default="tone")
    name_offset: str = Field(default="tone")
    start_time: float = Field(
        description="Start time of the event "
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event "
    )
    site_index: float = Field(
        description="Index identifier of the reward site."
    )


class Patch(BaseModel):
    """Mouse enters and exits a virtual patch."""

    name_onset: str = Field(default="patch_enter")
    name_offset: str = Field(default="patch_exit")
    start_time: float = Field(
        description="Start time of the event "
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event "
    )
    patch_index: float = Field(
        description="Index identifier of the patch."
    )


class Block(BaseModel):
    """Onset and offset of an experimental block."""

    name_onset: str = Field(default="block_onset")
    name_offset: str = Field(default="block_offset")
    start_time: float = Field(
        description="Start time of the event "
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event "
    )
    block_index: float = Field(
        description="Index identifier of the block."
    )


class InterSite(BaseModel):
    """Mouse traverses an inter-site interval."""

    name_onset: str = Field(default="inter_site_enter")
    name_offset: str = Field(default="inter_site_exit")
    start_time: float = Field(
        description="Start time of the event "
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event "
    )
    site_index: float = Field(
        description="Index identifier of the adjacent reward site."
    )


class InterPatch(BaseModel):
    """Mouse traverses an inter-patch interval."""

    name_onset: str = Field(default="inter_patch_enter")
    name_offset: str = Field(default="inter_patch_exit")
    start_time: float = Field(
        description="Start time of the event "
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event "
    )
    site_index: float = Field(
        description="Index identifier of the adjacent patch."
    )


class RewardProbability(BaseModel):
    """Reward probability sampled by the task logic."""

    name_onset: str = Field(default="reward_probability")
    name_offset: str = Field(default="reward_probability")
    start_time: float = Field(
        description="Start time of the event "
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event "
    )
    site_index: float = Field(
        description="Index identifier of the reward site."
    )


class Stop(BaseModel):
    """Mouse velocity crosses the stop threshold."""

    name_onset: str = Field(default="stop_onset")
    name_offset: str = Field(default="stop_offset")
    start_time: float = Field(
        description="Start time of the event "
    )
    stop_time: Optional[float] = Field(
        default=None,
        description="Stop time of the event "
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
            patch_index=i,
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
    stop_threshold: pd.DataFrame,
) -> list[Stop]:
    """Extract stop onset and offset events from the
    StopVelocityThreshold stream, where alternating rows
    represent stop onset and offset.

    Parameters
    ----------
    stop_threshold : pd.DataFrame
        StopVelocityThreshold dataframe with a 'data' column.

    Returns
    -------
    list[Stop]
        List of Stop events with start and stop times.
    """
    events = []

    timestamps = stop_threshold.index.values
    for i in range(0, len(timestamps) - 1, 2):
        events.append(Stop(
            start_time=timestamps[i],
            stop_time=timestamps[i + 1],
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

    odor_times = dataset.at("Behavior").at("HarpOlfactometer").load().at("EndValveState").load().data
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

    stop_threshold = dataset.at("Behavior").at("SoftwareEvents").at("StopVelocityThreshold").load().data
    stop_events = extract_stop_events(stop_threshold)
    for stop_event in stop_events:
        events.append(stop_event)

    return events

def sidecar_to_hed_dataframe(sidecar: dict) -> pd.DataFrame:
    """Build a HED reference dataframe from a BIDS sidecar JSON.

    Parameters
    ----------
    sidecar : dict
        BIDS sidecar JSON as a dictionary.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: value, meaning, hed_tag.
    """
    event_type = sidecar["event_type"]
    levels = event_type["Levels"]
    hed = event_type["HED"]

    # Build a lookup of placeholder -> HED identifier
    # e.g. site_index -> ID/#, reward_volume -> Param/volume-#
    placeholder_hed = {
        key: sidecar[key]["HED"]
        for key in sidecar
        if key not in ("event_type", "hed_defs")
        and "HED" in sidecar[key]
    }

    rows = []
    for event_name, meaning in levels.items():
        raw_hed = hed.get(event_name, "")

        # Replace placeholders like {site_index} with their HED identifier
        resolved_hed = raw_hed
        for placeholder, hed_id in placeholder_hed.items():
            resolved_hed = resolved_hed.replace(
                f"{{{placeholder}}}", hed_id
            )

        rows.append({
            "value": event_name,
            "meaning": meaning,
            "hed_tag": resolved_hed,
        })

    return pd.DataFrame(rows)

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
    # Infer number of channels from first event with concentration
    n_channels = next(
        (
            len(e.concentration)
            for e in events
            if hasattr(e, "concentration")
            and isinstance(e.concentration, list)
        ),
        0,
    )

    rows = []

    for event in events:
        concentration = getattr(event, "concentration", None)
        concentration_cols = {
            f"concentration_ch{i}": (
                concentration[i]
                if isinstance(concentration, list)
                else np.nan
            )
            for i in range(n_channels)
        }

        base = {
            "site_index": getattr(event, "site_index", np.nan),
            "patch_index": getattr(event, "patch_index", np.nan),
            "block_index": getattr(event, "block_index", np.nan),
            "reward_volume": getattr(event, "reward_volume", np.nan),
            **concentration_cols,
        }

        rows.append({
            "timestamp": event.start_time,
            "event_type": event.name_onset,
            **base,
        })

        if event.stop_time is not None:
            rows.append({
                "timestamp": event.stop_time,
                "event_type": event.name_offset,
                **base,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df