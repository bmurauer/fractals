from fractals.animation import ScalingAnimation, AttributeAnimation
from fractals.flame import Flame
from fractals.midi import MidiListener
from fractals.transitions import FunctionForm

flame = Flame.from_file("flames.flame", "057", draft=True)
# listener = MidiListener("/home/benjamin/documents/midi/midi-test.mid")
# total_frames = listener.get_max_frames()

flame.xform_animations = [
    AttributeAnimation(
        xform_index=0,
        attribute="julian",
        value_from=0.1,
        value_to=0.5,
        start_frame=0,
        end_frame=60,
        method=FunctionForm.LINEAR,
    ),
    AttributeAnimation(
        xform_index=0,
        attribute="julian",
        value_from=0.8,
        start_frame=30,
        end_frame=50,
        method=FunctionForm.SINUSOIDAL,
    ),
]

flames = flame.animate(total_frames=60)

flames.write_file()
# flames.render()
# flames.convert_to_movie()
