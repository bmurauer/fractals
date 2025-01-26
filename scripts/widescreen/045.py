from fractals.flame import Flame
from fractals.transitions import make_transition, FunctionForm
from fractals.xform import OrbitAnimation, RotationAnimation, loop

flame = Flame.from_file("flames.flame", "045", draft=True)
flame.add_palette_rotation_animation()

flame.xforms[0].animations = [
    OrbitAnimation(
        start_frame=0,
        animation_length=150,
        radius=0.05,
        reverse=False,
        transition=make_transition(FunctionForm.LINEAR),
    )
]
flame.xforms[1].animations = loop(
    RotationAnimation,
    animation_length=50,
    total_frames=150,
    value_to=1.0,
    reverse=False,
    transition=make_transition(FunctionForm.SIGMOID),
)

flames = flame.animate(total_frames=150)

# flames.write_file()
flames.render()
flames.convert_to_movie()
