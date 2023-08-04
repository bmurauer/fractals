from fractals.flame import Flame
from fractals.transform import Transform
from fractals.transitions import beating, make_transition, FunctionForm
from fractals.xform import (
    OrbitAnimation,
    AttributeAnimation,
    ScalingAnimation,
    RotationAnimation,
    TranslationAnimation,
    loop,
)

flame = Flame.from_file("tests/heartgrid.flame", draft=True)
flame.add_palette_rotation_animation()
flame.xforms[0].animations = loop(
    AttributeAnimation,
    animation_length=15,
    total_frames=120,
    offset=20,
    attribute="blur_heart",
    value_to=0.35,
    transition=make_transition(FunctionForm.SIGMOID),
)

flame.xforms[1].animations = loop(
    RotationAnimation,
    animation_length=40,
    total_frames=120,
    value_from=0.0,
    value_to=0.2,
    transition=make_transition(FunctionForm.INVERSE_SIGMOID),
    stack=True,
)
flames = flame.animate(total_frames=120)
flames.render()
flames.convert_to_movie()
