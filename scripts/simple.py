from fractals.animation import (
    RotationXformAnimation,
    loop,
    ScalingXformAnimation,
    AttributeXformAnimation,
    OrbitXformAnimation,
    TranslationXformAnimation,
)
from fractals.flame import Flame
from fractals.transform import Transform
from fractals.transitions import FunctionForm

flame: Flame = Flame.from_file("simple.flame", draft=True)
n_frames = 60

flame.xform_animations = [
    # *loop(
    #     RotationAnimation,
    #     xform_index=0,
    #     animation_length=100,
    #     total_frames=n_frames,
    #     value_from=0.0,
    #     value_to=1.0,
    #     debug=True,
    # )
    # ScalingAnimation(
    #     start_frame=0,
    #     end_frame=100,
    #     value_from=0.9,
    #     value_to=1.0,
    #     xform_index=0,
    #     debug=True,
    # ),
    # RotationAnimation(
    #     start_frame=0,
    #     end_frame=90,
    #     value_from=0.0,
    #     value_to=1.0,
    #     xform_index=0,
    # ),
    # RotationAnimation(
    #     start_frame=60,
    #     end_frame=150,
    #     value_from=0.0,
    #     value_to=1.0,
    #     xform_index=0,
    # ),
    # AttributeAnimation(
    #     attribute="gaussian_blur",
    #     xform_index=1,
    #     start_frame=100,
    #     end_frame=n_frames,
    #     value_from=0.0,
    #     value_to=0.1,
    #     method=FunctionForm.LINEAR,
    #     debug=True,
    # ),
    OrbitXformAnimation(
        radius=0.01,
        xform_index=1,
        start_frame=0,
        end_frame=60,
        value_from=0.0,
        value_to=1.0,
        method=FunctionForm.LINEAR,
        debug=True,
    )
    # TranslationAnimation(
    #     xform_index=1,
    #     start_frame=0,
    #     end_frame=60,
    #     value_from=Transform("0.5 0.2 -0.2 0.5 0 0"),
    #     value_to=Transform("0.75 0 -0.2 0.5 0 0"),
    #     method=FunctionForm.LINEAR,
    #     debug=True,
    # )
]

animation = flame.animate(total_frames=n_frames)
animation.render()
animation.convert_to_movie()
