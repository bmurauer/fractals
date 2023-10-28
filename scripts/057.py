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
    ScalingAnimation(
        start_frame=0,
        animation_length=15,
        value_from=1.1,
        value_to=1.0,
        transition=make_transition(FunctionForm.INVERSE_SIGMOID),
    ),
    ScalingAnimation(
        start_frame=30,
        animation_length=15,
        value_from=1.1,
        value_to=1.0,
        transition=make_transition(FunctionForm.INVERSE_SIGMOID),
    ),
]

flames = flame.animate(total_frames=60)

flames.write_file()
# flames.render()
# flames.convert_to_movie()
