from fractals.flame import Flame
from fractals.transitions import pulsating, make_transition, FunctionForm
from fractals.xform import AttributeAnimation, loop

flame = Flame.from_file("flames.flame", "032", draft=True)


flame.xforms[2].animations = [
    AttributeAnimation(
        start_frame=0,
        animation_length=120,
        attribute="ztranslate",
        value_to=0.217,
        transition=make_transition(FunctionForm.SINUSOIDAL),
    ),
    AttributeAnimation(
        start_frame=120,
        animation_length=240,
        attribute="ztranslate",
        value_to=0.217,
        reverse=True,
        transition=make_transition(FunctionForm.SINUSOIDAL),
    ),
    *loop(
        AttributeAnimation,
        animation_length=60,
        total_frames=240,
        attribute="cylinder",
        value_from=0,
        value_to=0.05,
        transition=make_transition(FunctionForm.INVERSE_SIGMOID),
    ),
]
flames = flame.animate(total_frames=240)
flames.render()
flames.convert_to_movie()
