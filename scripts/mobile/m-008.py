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
label = "Crystal"
FPS = 60
SECONDS = 15
draft = True
N_FRAMES = FPS * 3 if draft else FPS * SECONDS

flame = Flame.from_file(
    file_name="mobile-flames.flame", flame_name=name, draft=draft
)

flame.flame_animations = [
    PaletteRotationFlameAnimation(
        start_frame=0,
        end_frame=N_FRAMES,
        value_from=0,
        value_to=-1,
        method=FunctionForm.INVERSE_SIGMOID,
    ),
]

flame.xform_animations = [
    OrbitXformAnimation(
        xform_index=1,
        start_frame=0,
        end_frame=N_FRAMES,
        radius=0.02,
        value_from=0,
        value_to=1.0,
        method=FunctionForm.LINEAR,
    ),
    # OrbitXformAnimation(
    #     xform_index=1,
    #     start_frame=0,
    #     end_frame=N_FRAMES,
    #     radius=0.1,
    #     value_from=0,
    #     value_to=1.0,
    #     method=FunctionForm.LINEAR,
    # ),
    # OrbitXformAnimation(
    #     xform_index=2,
    #     start_frame=0,
    #     end_frame=N_FRAMES,
    #     radius=0.1,
    #     value_from=0,
    #     value_to=1.0,
    #     method=FunctionForm.LINEAR,
    # ),
]

flames = flame.animate(
    total_frames=N_FRAMES, directory_name=f"rendered-{name}"
)
flames.render()
flames.convert_to_movie(crf=25)
