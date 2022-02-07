from fractals.flame import Flame
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("flames.flame", "048")
flame = Flame.from_element(xml)
flame.add_palette_rotation_animation()
flame.xforms[0].add_attr_animation("elliptic", target=1.15)
flame.xforms[1].add_attr_animation("cot", target=-2.2, n_repetitions=3)
flame.xforms[1].add_attr_animation("cotq", target=0.9, n_repetitions=2)
flames = flame.animate(total_frames=3000)

# flames.write_file()
flames.render()
flames.convert_to_movie()
