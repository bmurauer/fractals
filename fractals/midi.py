import math
import sys
from abc import ABC, abstractmethod
from typing import Optional, Dict, Type

from mido import MidiFile

from logzero import logger


def find_bpm(mid: MidiFile) -> int:
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            return 60_000_000 / msg.tempo


def set_absolute_times(mid: MidiFile) -> int:
    track_counts = []
    for track in mid.tracks:
        count = 0
        for event in track:
            count += event.time
            event.time = count
        track_counts.append(count)
    return max(track_counts)


class MidiListener:
    def __init__(self, path_to_midi: str, fps: int = 30):
        self.mid: MidiFile = MidiFile(path_to_midi)
        self.n_ticks = set_absolute_times(self.mid)

        self.fps = fps
        bpm = find_bpm(self.mid)
        beat_duration = 60 / bpm
        tpb = int(self.mid.ticks_per_beat)
        n_beats = math.ceil(self.n_ticks // tpb) + 1

        self.max_ticks = n_beats * tpb
        max_seconds = n_beats * beat_duration
        self.tick_duration = max_seconds / self.max_ticks

        logger.info(
            "read midi file with: bpm=%d, tpb=%d, n_beats=%d",
            bpm,
            tpb,
            n_beats,
        )
        logger.info(
            "will create %d frames @ %d fps, resulting in %f " "seconds",
            self.get_max_frames(),
            self.fps,
            self.get_max_frames() / self.fps,
        )
        # print(self.mid)

    def get_max_frames(self) -> int:
        return round(self.tick_to_frame(self.max_ticks))

    def tick_to_frame(self, tick):
        return tick * self.tick_duration * self.fps

    def loop(
        self,
        animation_class: Type,
        filter_track_name: Optional[str] = None,
        channel: Optional[int] = None,
        filter_note: Optional[int] = None,
        trigger: Optional[str] = "note_on",
        value_from: float = 0.0,
        amount: float = 0.1,
        animation_length: Optional[int] = None,
        **kwargs,
    ):
        if "value_to" in kwargs:
            raise Exception("in 'loop' mode, 'value_to' is not supported.")
        all_frames = self.get_animation_frames(
            filter_track_name, channel, filter_note, trigger
        )
        result = []
        for idx, frame in enumerate(all_frames):
            value_to = value_from + amount
            start_frame = frame
            if idx + 1 < len(all_frames):
                end_frame = all_frames[idx + 1]
            else:
                end_frame = self.get_max_frames()
            logger.debug(
                "adding %s animation frames: %d - %d, values: %f - %f",
                animation_class.__class__.__name__,
                start_frame,
                end_frame,
                value_from,
                value_to,
            )
            result.append(
                animation_class(
                    start_frame=start_frame,
                    animation_length=animation_length
                    or end_frame - start_frame,
                    value_from=value_from,
                    value_to=value_to,
                    **kwargs,
                )
            )
            value_from = value_to
        return result

    def iterate(
        self,
        animation_class: Type,
        filter_track_name: Optional[str] = None,
        channel: Optional[int] = None,
        filter_note: Optional[int] = None,
        trigger: Optional[str] = "note_on",
        amount: float = 0.1,
        **kwargs,
    ):
        if "animation_length" not in kwargs:
            raise Exception(
                "in 'iterate' mode, 'animation_length' is " "required."
            )
        all_frames = self.get_animation_frames(
            filter_track_name, channel, filter_note, trigger
        )
        result = []
        for frame in all_frames:
            result.append(animation_class(start_frame=frame, **kwargs))
        return result

    def get_animation_frames(
        self,
        filter_track_name: Optional[str] = None,
        channel: Optional[int] = None,
        filter_note: Optional[int] = None,
        trigger: Optional[str] = "note_on",
    ):
        animations = []
        for track in self.mid.tracks:
            if (
                filter_track_name is not None
            ) and filter_track_name != track.name:
                continue
            for note in track:
                if channel is not None and channel != note.channel:
                    continue
                if trigger is not None and trigger != note.type:
                    continue
                if filter_note is not None and (filter_note != note.note):
                    continue
                if note.type == "note_on" and note.velocity == 0:
                    continue
                animations.append(self.tick_to_frame(note.time))
        return animations


#
# from fractals.xform import ScalingAnimation
#
# MidiListener(
#     "/home/benjamin/documents/midi/midi-test.mid"
# ).generate_animations(ScalingAnimation, filter_track_name="Snare Drum")
