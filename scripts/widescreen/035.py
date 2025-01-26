from fractals.flame import Flame
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("flames.flame", "035")
flame = Flame.from_element(xml)
flame.add_palette_rotation_animation()
flame.xforms[3].add_attr_animation("elliptic", 1.1)
flame.xforms[3].add_orbit_animation(radius=0.1)
flame.xforms[4].add_orbit_animation(radius=0.75, n_repetitions=2)
flames = flame.animate(total_frames=1500)
flames.write_file()
flames.render()
flames.convert_to_movie()
