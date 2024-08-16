from fractals.animation import (
    PaletteRotationFlameAnimation,
)
from fractals.animation import (
    OrbitXformAnimation,
    AttributeXformAnimation,
    bounce,
)
from fractals.flame import Flame
from fractals.transitions import FunctionForm

n_frames = 3600

flame = Flame.from_file("mobile-flames.flame", "001", draft=False)

flame.flame_animations = [
    PaletteRotationFlameAnimation(
        start_frame=0,
        end_frame=n_frames,
        value_from=0,
        value_to=1.0,
        method=FunctionForm.LINEAR,
    ),
]

flame.xform_animations = [
    OrbitXformAnimation(
        radius=0.1,
        xform_index=0,
        start_frame=0,
        end_frame=n_frames,
        value_from=0.0,
        value_to=1.0,
        method=FunctionForm.LINEAR,
    ),
    OrbitXformAnimation(
        radius=0.1,
        xform_index=1,
        start_frame=0,
        end_frame=n_frames,
        value_from=0.0,
        value_to=-1.0,
        method=FunctionForm.LINEAR,
    ),
    OrbitXformAnimation(
        radius=0.1,
        xform_index=2,
        start_frame=0,
        end_frame=n_frames,
        value_from=0.0,
        value_to=1.0,
        method=FunctionForm.LINEAR,
    ),
]

flames = flame.animate(total_frames=n_frames)
flames.render()
flames.convert_to_movie()
