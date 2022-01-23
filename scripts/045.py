from fractals.flame import Flame
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("flames.flame", "045")
flame = Flame.from_element(xml)
flame.add_palette_rotation_animation()
flame.xforms[0].add_orbit_animation(0.05)
flame.xforms[1].add_rotation_animation(n_rotations=-1)
flames = flame.animate(total_frames=150)

flames.write_file()
# flames.render()
# flames.convert_to_movie()
