from fractals.flame import Flame
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("flames.flame", "036")
flame = Flame.from_element(xml)
flame.add_palette_rotation_animation()
flame.xforms[2].add_attr_animation("elliptic", 0.9)
flame.xforms[2].add_attr_animation("cylinder2", 0.5)
flame.xforms[4].add_rotation_animation()
flames = flame.animate(total_frames=1500)
flames.write_file()
flames.render()
flames.convert_to_movie()
