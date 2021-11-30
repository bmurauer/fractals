from fractals.flame import Flame
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("flames.flame", "037_Peacock")
flame = Flame.from_element(xml)
flame.add_palette_rotation_animation()
flame.xforms[2].add_rotation_animation()
flame.xforms[2].add_orbit_animation(radius=0.5)
flames = flame.animate(total_frames=1500)
flames.write_file()
flames.render()
flames.convert_to_movie()
