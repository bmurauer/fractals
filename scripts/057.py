from fractals.flame import Flame
from fractals.midi import MidiListener
from fractals.transitions import make_transition, FunctionForm
from fractals.xform import (
    ScalingAnimation,
    loop,
    AttributeAnimation,
    RotationAnimation,
)

flame = Flame.from_file("flames.flame", "057", draft=True)
listener = MidiListener("/home/benjamin/documents/midi/midi-test.mid")
total_frames = listener.get_max_frames()

scaling_kwargs = dict(
    value_to=1.0,
    transition=make_transition(FunctionForm.INVERSE_SIGMOID),
)

flame.xforms[0].animations = [
    *listener.iterate(
        ScalingAnimation,
        filter_track_name="Snare Drum",
        value_from=1.1,
        animation_length=15,
        **scaling_kwargs
    ),
]
flame.xforms[2].animations = [
    *listener.iterate(
        ScalingAnimation,
        filter_track_name="Timpani",
        value_from=1.1,
        animation_length=24,
        **scaling_kwargs
    ),
    *listener.iterate(
        AttributeAnimation,
        attribute="julian",
        animation_length=15,
        filter_track_name="Snare Drum",
        value_from=1.1,
        value_to=1.0,
        transition=make_transition(FunctionForm.INVERSE_SIGMOID),
    ),
]
flame.xforms[1].animations = [
    *listener.iterate(
        ScalingAnimation,
        filter_track_name="Bass Drum",
        value_from=1.4,
        animation_length=90,
        **scaling_kwargs
    ),
    AttributeAnimation(
        attribute="julian",
        start_frame=0,
        animation_length=480,
        value_from=0.6,
        value_to=1.65,
        transition=make_transition(FunctionForm.LINEAR),
    ),
]

flame.xforms[3].animations = listener.loop(
    RotationAnimation,
    amount=0.02,
    value_from=0.0,
    filter_track_name="Timpani",
    transition=make_transition(FunctionForm.INVERSE_SIGMOID),
)

flames = flame.animate(total_frames=total_frames)

# flames.write_file()
flames.render()
flames.convert_to_movie()
