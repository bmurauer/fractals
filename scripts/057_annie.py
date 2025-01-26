import math

from fractals.flame import Flame
from fractals.midi import MidiListener
from fractals.transitions import FunctionForm
from fractals.animation import (
    ScalingXformAnimation,
    RotationXformAnimation,
    OrbitXformAnimation,
    loop,
)

flame = Flame.from_file("flames.flame", "057", draft=True)
listener = MidiListener("./midi/Walz for Annie.mid")
# n_frames = listener.get_max_frames()
n_frames = 210

flame.xform_animations = [
    *listener.iterate(
        xform_index=0,
        animation_class=ScalingXformAnimation,
        filter_track_name="Snare Drum",
        animation_length=25,
        value_from=1.05,
        value_to=1.0,
        method=FunctionForm.INVERSE_SIGMOID,
    ),
    *listener.iterate(
        xform_index=1,
        animation_class=ScalingXformAnimation,
        filter_track_name="Bass Drum",
        animation_length=35,
        value_from=1.1,
        value_to=1.0,
        method=FunctionForm.INVERSE_SIGMOID,
    ),
    # *listener.loop(
    #     xform_index=2,
    #     animation_class=RotationAnimation,
    #     filter_track_name="Piano",
    #     amount=0.1,
    #     animation_length=30,
    #     method=FunctionForm.LINEAR,
    # ),
    # *loop(
    #     OrbitAnimation,
    #     xform_index=2,
    #     method=FunctionForm.LINEAR,
    #     animation_length=50,
    #     total_frames=n_frames,
    #     value_from=0.0,
    #     value_to=1.0,
    #     radius=0.05,
    # ),
    # *loop(
    #     RotationAnimation,
    #     xform_index=3,
    #     method=FunctionForm.LINEAR,
    #     animation_length=400,
    #     total_frames=n_frames,
    #     value_from=0.0,
    #     value_to=1.0,
    # ),
    # *loop(
    #     OrbitAnimation,
    #     xform_index=3,
    #     method=FunctionForm.LINEAR,
    #     animation_length=180,
    #     total_frames=n_frames,
    #     value_from=0.0,
    #     value_to=-1.0,
    #     radius=0.15,
    # ),
]
flames = flame.animate(total_frames=n_frames)
# flames.write_file()
flames.render()
flames.convert_to_movie()
