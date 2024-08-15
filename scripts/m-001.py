from fractals.flameanimation import (
    FlameAnimation,
    AttributeFlameAnimation,
    PaletteRotationFlameAnimation,
)
from fractals.xformanimation import (
    OrbitXformAnimation,
    AttributeXformAnimation,
)
from fractals.flame import Flame
from fractals.transitions import FunctionForm

n_frames = 180

flame = Flame.from_file("mobile-flames.flame", "001", draft=True)

flame.flame_animations = [
    PaletteRotationFlameAnimation(
        start_frame=0,
        end_frame=n_frames,
        value_from=0,
        value_to=1,
        method=FunctionForm.LINEAR,
    ),
]

flame.xform_animations = [
    AttributeXformAnimation(
        xform_index=0,
        start_frame=0,
        end_frame=n_frames // 2,
        value_from=1.0,
        value_to=0.95,
        attribute="linear",
        method=FunctionForm.SINUSOIDAL,
    ),
    AttributeXformAnimation(
        xform_index=0,
        start_frame=n_frames // 2 + 1,
        end_frame=n_frames,
        value_from=0.95,
        value_to=1.0,
        attribute="linear",
        method=FunctionForm.SINUSOIDAL,
    ),
    AttributeXformAnimation(
        xform_index=2,
        start_frame=0,
        end_frame=n_frames // 2,
        value_from=0.5,
        value_to=0.6,
        attribute="butterfly",
        method=FunctionForm.SINUSOIDAL,
    ),
    AttributeXformAnimation(
        xform_index=2,
        start_frame=n_frames // 2 + 1,
        end_frame=n_frames,
        value_from=0.6,
        value_to=0.5,
        attribute="butterfly",
        method=FunctionForm.SINUSOIDAL,
    ),
    OrbitXformAnimation(
        radius=0.001,
        xform_index=2,
        start_frame=0,
        end_frame=n_frames,
        value_from=0.0,
        value_to=1.0,
        method=FunctionForm.LINEAR,
    ),
]

flames = flame.animate(total_frames=n_frames)

flames.write_file()
flames.render()
flames.convert_to_movie()
