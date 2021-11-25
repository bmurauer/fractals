from fractals.flame import Flame
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("flames.flame", "031")
flame = Flame.from_element(xml)
flame.xforms[0].add_attr_animation("elliptic", 1.1)
flames = flame.animate(total_frames=1500)
flames.write_file()
flames.render()
flames.convert_to_movie()
