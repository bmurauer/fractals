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

FPS = 60

draft = False
N_FRAMES = FPS * 3 if draft else FPS * 15

flame = Flame.from_file(
    file_name="mobile-flames.flame", flame_name="004", draft=draft
)

flame.flame_animations = [
    PaletteRotationFlameAnimation(
        start_frame=0,
        end_frame=N_FRAMES,
        value_from=0,
        value_to=1.0,
        method=FunctionForm.LINEAR,
    ),
]

flame.xform_animations = [
    OrbitXformAnimation(
        xform_index=0,
        radius=1.0,
        start_frame=0,
        end_frame=N_FRAMES,
        value_from=0.0,
        value_to=1.0,
        method=FunctionForm.LINEAR,
    ),
    *bounce(
        AttributeXformAnimation,
        xform_index=1,
        attribute="fisheye",
        start_frame=0,
        end_frame=N_FRAMES,
        value_from=0.0,
        value_to=0.3,
    ),
]

flames = flame.animate(total_frames=N_FRAMES, directory_name="rendered-004")
flames.render()
# flames.convert_to_movie(crf=25)
