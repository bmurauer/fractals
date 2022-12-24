from fractals.flame import Flame, Transform
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("simple.flame")
flame = Flame.from_element(xml, draft=True)
# flame.add_palette_rotation_animation()
# flame.xforms[2].add_rotation_animation(2, 120, bumpyness=1)
flame.xforms[2].add_scale_animation(1.2, n_repetitions=1, bpm=120)
# flame.xforms[2].add_translation_animation(Transform("0 0 1.0 0 0 1.0"), bpm=60)
flames = flame.animate(total_frames=240)

# flames.write_file()
flames.render()
flames.convert_to_movie()
