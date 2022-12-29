from fractals.flame import Flame
from fractals.transitions import beating_up_down, pulsating

flame: Flame = Flame.from_file("simple.flame", draft=True)
flame.add_palette_rotation_animation()
# flame.xforms[0].add_rotation_animation(1, bpm=60)
flame.xforms[1].add_scale_animation(1.2, envelope=pulsating, bpm=120)
# flame.xforms[1].add_translation_animation(Transform("0 0 1.0 0 0 1.0"), bpm=90)
# flame.xforms[2].add_attr_animation(attribute="spherical", target=0.1, bpm=90)
# flame.xforms[2].add_orbit_animation(0.1, bpm=120)
animation = flame.animate(total_frames=240)
# animation.one_file_per_flame = True
# flames.write_file()
animation.render()
animation.convert_to_movie()
