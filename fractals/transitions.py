import sys
from typing import Union

from fractals.transform import Transform
from fractals.utils import (
    logger,
    ramp_inverse_sigmoid,
    ramp_linear,
    ramp_sigmoid,
    ramp_sinusoidal,
)

FPS = 30


def smooth(
    frame: int,
    total_frames: int,
    value_from: Union[Transform, float],
    value_to: Union[Transform, float],
    method: str,
    method_bumpyness: float = 0.5,
) -> float:
    factor = None
    if method == "linear":
        factor = ramp_linear(frame, total_frames)
    elif method == "sinusoidal":
        factor = ramp_sinusoidal(frame, total_frames)
    elif method == "sigmoid":
        factor = ramp_sigmoid(frame, total_frames, method_bumpyness)
    elif method == "inverse_sigmoid":
        factor = ramp_inverse_sigmoid(frame, total_frames, method_bumpyness)
    else:
        raise Exception("unknown method: " + method)
    return value_from + (value_to - value_from) * factor


def pulsating(
    frame: int,
    total_frames: int,
    value_from: Union[Transform, float],
    value_to: Union[Transform, float],
    bpm: float,
    beat_bumpyness: float = 0.5,
    **kwargs
):
    frames_per_beat = FPS / (bpm / 60)
    if total_frames % frames_per_beat != 0:
        logger.error(
            "Can't subdivide %d frames into beats of %d frames (bpm=%d, FPS=%s)",
            total_frames,
            frames_per_beat,
            bpm,
            FPS,
        )
        sys.exit(1)
    n_beats: int = total_frames // frames_per_beat
    # we are in this beat right now:
    current_beat = frame // frames_per_beat
    # within the beat, we are currently at frame:
    frame_in_beat = frame % frames_per_beat
    return smooth(
        frame=frame_in_beat,
        total_frames=frames_per_beat,
        value_from=value_to,
        value_to=value_from,
        method="inverse_sigmoid",
    )


def beating(
    frame: int,
    total_frames: int,
    value_from: Union[Transform, float],
    value_to: Union[Transform, float],
    method: str,
    method_bumpyness: float = 0.5,
    bpm: float = None,
    beat_bumpyness=0.5,
) -> float:

    if bpm is None or beat_bumpyness == 0:
        return smooth(
            frame=frame,
            total_frames=total_frames,
            value_from=value_from,
            value_to=value_to,
            method=method,
            method_bumpyness=method_bumpyness,
        )

    frames_per_beat = FPS / (bpm / 60)
    if total_frames % frames_per_beat != 0:
        logger.error(
            "Can't subdivide %d frames into beats of %d frames (bpm=%d, FPS=%s)",
            total_frames,
            frames_per_beat,
            bpm,
            FPS,
        )
        sys.exit(1)
    n_beats: int = total_frames // frames_per_beat
    # we are in this beat right now:
    current_beat = frame // frames_per_beat
    # within the beat, we are currently at frame:
    frame_in_beat = frame % frames_per_beat
    beat_start_frame = current_beat * frames_per_beat
    beat_end_frame = (current_beat + 1) * frames_per_beat - 1
    start_value = smooth(
        frame=beat_start_frame,
        total_frames=total_frames,
        value_from=0,
        value_to=1,
        method=method,
        method_bumpyness=method_bumpyness,
    )
    end_value = smooth(
        frame=beat_end_frame,
        total_frames=total_frames,
        value_from=0,
        value_to=1,
        method=method,
        method_bumpyness=method_bumpyness,
    )
    bumpy_factor = smooth(
        frame=frame_in_beat,
        total_frames=frames_per_beat,
        value_from=start_value,
        value_to=end_value,
        method="inverse_sigmoid",
        method_bumpyness=beat_bumpyness,
    )
    return value_from + (value_to - value_from) * bumpy_factor


def repeat(
    n_repetitions: int, frame: int, total_frames: int, envelope: callable, **kwargs
) -> float:
    if total_frames % n_repetitions != 0:
        raise Exception(
            "Can't subdivide %d total frames into %d repetitions.",
            total_frames,
            n_repetitions,
        )
    frames_per_repetition = total_frames / n_repetitions
    frame_in_repetition = frame % frames_per_repetition
    return envelope(
        frame=frame_in_repetition, total_frames=frames_per_repetition, **kwargs
    )


def beating_up_down(
    frame: int,
    total_frames: int,
    value_from: Union[Transform, float],
    value_to: Union[Transform, float],
    **kwargs
) -> float:
    frames_per_half = total_frames // 2
    frame_in_half = frame % frames_per_half
    if frame < frames_per_half:
        return beating(
            frame=frame_in_half,
            total_frames=frames_per_half,
            value_from=value_from,
            value_to=value_to,
            **kwargs,
        )
    else:
        return beating(
            frame=frame_in_half,
            total_frames=frames_per_half,
            value_from=value_to,
            value_to=value_from,
            **kwargs,
        )
