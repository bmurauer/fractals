from fractals.flame import Flame
from fractals.transform import Transform
from fractals.transitions import beating, make_transition, FunctionForm
from fractals.xform import (
    OrbitAnimation,
    AttributeAnimation,
    ScalingAnimation,
    RotationAnimation,
    TranslationAnimation,
)

flame = Flame.from_file("tests/heartgrid.flame")
flame.draft = True
# flame.xforms[0].animations = [
#     OrbitAnimation(
#         radius=1.0,
#         start_frame=10,
#         animation_length=10,
#         transition=make_transition(FunctionForm.INVERSE_SIGMOID),
#     ),
#     AttributeAnimation(
#         start_frame=15,
#         animation_length=10,
#         transition=make_transition(FunctionForm.LINEAR),
#         attribute="elliptic",
#         target=1.1,
#     ),
# ]
flame.xforms[1].animations = [
    TranslationAnimation(
        start_frame=0,
        animation_length=10,
        transition=make_transition(FunctionForm.LINEAR),
        target_transform=Transform("1.0 0 0 1.0 -1.0 0"),
    ),
    TranslationAnimation(
        start_frame=10,
        animation_length=10,
        reverse=True,
        transition=make_transition(FunctionForm.SIGMOID),
        target_transform=Transform("1.0 0 0 1.0 -1.0 0"),
    ),
]
# flame.xforms[2].animations = [
#     RotationAnimation(
#         target=1.0,
#         start_frame=10,
#         animation_length=10,
#         transition=make_transition(FunctionForm.INVERSE_SIGMOID),
#     ),
# ]

# flame.xforms[0].add_orbit_animation(radius=1.0)
# flame.xforms[0].add_rotation_animation(2)
# flame.xforms[0].add_scale_animation(2)
flames = flame.animate(total_frames=30)
flames.write_file()
