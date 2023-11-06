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
listener = MidiListener("./midi/Walz for Annie.mid")

scaling_kwargs = dict(
    value_to=1.0,
    transition=make_transition(FunctionForm.INVERSE_SIGMOID),
)

flame.xforms[0].animations = [
    *listener.iterate(
        animation_class=ScalingAnimation,
        filter_track_name="Snare Drum",
        animation_length=15,
        value_from=1.1,
        **scaling_kwargs
    )
]
flame.xforms[1].animations = [
    *listener.iterate(
        animation_class=ScalingAnimation,
        filter_track_name="Bass Drum",
        animation_length=15,
        value_from=1.3,
        **scaling_kwargs
    )
]
flame.xforms[2].animations = [
    *listener.loop(
        animation_class=RotationAnimation,
        filter_track_name="Piano",
        amount=0.1,
        animation_length=15,
        transition=make_transition(FunctionForm.INVERSE_SIGMOID),
    )
]

flames = flame.animate(total_frames=400)
# flames = flame.animate(total_frames=listener.get_max_frames())

# flames.write_file()

flames.render()
flames.convert_to_movie()
