from fractals.flame import Flame
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("tests/heartgrid.flame")
flame = Flame.from_element(xml)
flame.xforms[0].add_orbit_animation(radius=1.0)
flame.xforms[0].add_rotation_animation(2)
flame.xforms[0].add_scale_animation(2)
flames = flame.animate(n_frames=10)
flames.write_file()
