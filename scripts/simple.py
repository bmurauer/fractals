from fractals.flame import Flame, Transform
from fractals.utils import get_flame_from_file

flame = Flame.from_file("simple.flame", draft=True)
flame.add_palette_rotation_animation()
# flame.xforms[0].add_rotation_animation(2)
flame.xforms[1].add_scale_animation(2.0)
# flame.xforms[1].add_translation_animation(Transform("0 0 1.0 0 0 1.0"), bpm=90)
# flame.xforms[2].add_attr_animation(attribute="spherical", target=0.1, bpm=90)
# flame.xforms[2].add_orbit_animation(0.1, bpm=120)
flames = flame.animate(total_frames=240)

# flames.write_file()
flames.render()
flames.convert_to_movie()
