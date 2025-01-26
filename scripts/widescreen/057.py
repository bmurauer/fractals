from fractals.animation import (
    ScalingXformAnimation,
    AttributeXformAnimation,
)
from fractals.flame import Flame
from fractals.midi import MidiListener
from fractals.transitions import FunctionForm

flame = Flame.from_file("flames.flame", "057", draft=True)
# listener = MidiListener("/home/benjamin/documents/midi/midi-test.mid")
# total_frames = listener.get_max_frames()

flame.xform_animations = [
    AttributeXformAnimation(
        xform_index=0,
        attribute="julian",
        value_from=0.1,
        value_to=0.5,
        start_frame=0,
        end_frame=120,
        method=FunctionForm.LINEAR,
    ),
    AttributeXformAnimation(
        xform_index=0,
        attribute="julian",
        value_from=0.8,
        start_frame=80,
        end_frame=100,
        method=FunctionForm.SINUSOIDAL,
    ),
]

flames = flame.animate(total_frames=120)

flames.write_file()
flames.render()
flames.convert_to_movie()
