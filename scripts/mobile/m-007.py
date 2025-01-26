from fractals.animation import (
    PaletteRotationFlameAnimation,
    RotationXformAnimation,
)
from fractals.animation import (
    OrbitXformAnimation,
    AttributeXformAnimation,
    bounce,
)
from fractals.flame import Flame
from fractals.transitions import FunctionForm
from fractals.video import Video

name = "007"
label = "Earth"
FPS = 1
SECONDS = 10
draft = False

N_FRAMES = 3 * SECONDS if draft else FPS * SECONDS

flame = Flame.from_file(
    file_name="mobile-flames.flame", flame_name=name, draft=draft
)

flame.flame_animations = [
    PaletteRotationFlameAnimation(
        start_frame=0,
        end_frame=N_FRAMES,
        value_from=0,
        value_to=-1,
        method=FunctionForm.LINEAR,
    ),
]

flame.xform_animations = [
    OrbitXformAnimation(
        xform_index=1,
        start_frame=0,
        end_frame=N_FRAMES,
        radius=0.1,
        value_from=0,
        value_to=1.0,
        method=FunctionForm.LINEAR,
    ),
    RotationXformAnimation(
        xform_index=1,
        start_frame=0,
        end_frame=N_FRAMES,
        value_from=0,
        value_to=1.0,
        method=FunctionForm.LINEAR,
    ),
    *bounce(
        RotationXformAnimation,
        xform_index=0,
        start_frame=0,
        end_frame=N_FRAMES,
        value_from=0,
        value_to=-0.1,
        method=FunctionForm.SINUSOIDAL,
    ),
]

flames = Video.from_animation(flame, N_FRAMES, draft=True)
flames.deploy()

print(flames)
