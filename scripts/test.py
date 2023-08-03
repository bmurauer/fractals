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
flame.xforms[0].animations = loop(
    AttributeAnimation,
    animation_length=15,
    total_frames=120,
    offset=20,
    attribute="blur_heart",
    value_to=0.35,
    transition=make_transition(FunctionForm.SIGMOID),
    reverse=True,
    stack=False,
)

flame.xforms[1].animations = loop(
    RotationAnimation,
    animation_length=40,
    total_frames=120,
    value_from=0.0,
    value_to=0.2,
    reverse=False,
    transition=make_transition(FunctionForm.INVERSE_SIGMOID),
    stack=True,
)

# flame.xforms[1].animations = [
#     RotationAnimation(
#         start_frame=0,
#         animation_length=30,
#         value_from=0,
#         value_to=0.25,
#         reverse=False,
#         transition=make_transition(FunctionForm.INVERSE_SIGMOID),
#     ),
#     RotationAnimation(
#         start_frame=30,
#         animation_length=30,
#         value_from=0.25,
#         value_to=0.5,
#         reverse=False,
#         transition=make_transition(FunctionForm.INVERSE_SIGMOID),
#     ),
#     RotationAnimation(
#         start_frame=60,
#         animation_length=30,
#         value_from=0.5,
#         value_to=0.75,
#         reverse=False,
#         transition=make_transition(FunctionForm.INVERSE_SIGMOID),
#     ),
#     RotationAnimation(
#         start_frame=90,
#         animation_length=30,
#         value_from=0.75,
#         value_to=1,
#         reverse=False,
#         transition=make_transition(FunctionForm.INVERSE_SIGMOID),
#     ),
# ]

# flame.xforms[0].add_rotation_animation(2)
# flame.xforms[0].add_scale_animation(2)
flames = flame.animate(total_frames=120)
flames.render()
flames.convert_to_movie()
