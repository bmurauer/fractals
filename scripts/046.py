from fractals.flame import Flame
from fractals.transitions import make_transition, FunctionForm
from fractals.xform import (
    OrbitAnimation,
    RotationAnimation,
    loop,
    AttributeAnimation,
)

flame = Flame.from_file("flames.flame", "046", draft=True)
flame.add_palette_rotation_animation()

flame.xforms[0].animations = [
    AttributeAnimation(
        start_frame=0,
        animation_length=240,
        attribute="bubbleT3D",
        value_to=0.25,
        transition=make_transition(FunctionForm.SINUSOIDAL),
    )
]
flame.xforms[1].animations = [
    AttributeAnimation(
        attribute="julian",
        start_frame=0,
        animation_length=240,
        value_to=1.25,
        transition=make_transition(FunctionForm.SINUSOIDAL),
    )
]
flame.xforms[2].animations = [
    AttributeAnimation(
        attribute="julian",
        start_frame=0,
        animation_length=240,
        value_to=0.15,
        transition=make_transition(FunctionForm.SINUSOIDAL),
    )
]


flames = flame.animate(total_frames=240)

# flames.write_file()
flames.render()
flames.convert_to_movie()
