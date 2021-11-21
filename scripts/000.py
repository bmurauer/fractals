from fractals.flame import Flame
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("flames.flame", "000")
flame = Flame.from_element(xml)
flame.add_palette_rotation_animation()
flame.xforms[0].add_orbit_animation(radius=0.01)
flame.xforms[1].add_orbit_animation(radius=0.01)
flames = flame.animate(total_frames=10)
flames.write_file()
flames.render()
flames.convert_to_movie()
