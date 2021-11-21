from fractals.flame import Flame
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("flames.flame", "026")
flame = Flame.from_element(xml)
flame.xforms[0].add_orbit_animation(radius=0.5)
flame.palette.n_rotations = 1
flames = flame.animate(total_frames=1000)
flames.write_file()
flames.render()
flames.convert_to_movie()
