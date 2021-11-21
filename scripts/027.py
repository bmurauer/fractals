from fractals.flame import Flame
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("flames.flame", "027")
flame = Flame.from_element(xml)
flame.add_palette_rotation_animation()
flame.xforms[0].add_rotation_animation()
flame.xforms[0].add_scale_animation(factor=1.5, n_repetitions=2)
flame.xforms[2].add_rotation_animation(n_rotations=2)
flame.xforms[2].add_attr_animation("julian", 1.2)
flames = flame.animate(total_frames=1000)
flames.write_file()
flames.render()
flames.convert_to_movie()
