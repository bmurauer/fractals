from fractals.flame import Flame
from fractals.transitions import make_transition, FunctionForm
from fractals.xform import AttributeAnimation

flame = Flame.from_file("flames.flame", "048", draft=True)
flame.add_palette_rotation_animation()

flame.xforms[0].animations = [
    AttributeAnimation(
        start_frame=0,
        animation_length=20,
        value_to=1.15,
        attribute="elliptic",
        transition=make_transition(FunctionForm.SIGMOID),
    )
]

flame.xforms[1].animations = [
    AttributeAnimation(
        start_frame=0,
        animation_length=150,
        value_to=2.2,
        attribute="cot",
        transition=make_transition(FunctionForm.SIGMOID),
    ),
    AttributeAnimation(
        start_frame=150,
        animation_length=150,
        value_to=2.2,
        attribute="cot",
        transition=make_transition(FunctionForm.SIGMOID),
    ),
    AttributeAnimation(
        start_frame=0,
        animation_length=150,
        value_to=0.9,
        attribute="cotq",
        transition=make_transition(FunctionForm.SIGMOID),
    ),
    AttributeAnimation(
        start_frame=150,
        animation_length=150,
        value_to=0.9,
        attribute="cotq",
        transition=make_transition(FunctionForm.SIGMOID),
    ),
]

flames = flame.animate(total_frames=300)

# flames.write_file()
flames.render()
flames.convert_to_movie()
