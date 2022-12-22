from fractals.flame import Flame
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("simple.flame")
flame = Flame.from_element(xml, draft=True)
flame.add_palette_rotation_animation()
# flame.xforms[2].add_rotation_animation(1, 120, bumpyness=0.25)
flame.xforms[2].add_scale_animation(1.2, n_repetitions=4)
flames = flame.animate(total_frames=300)

# flames.write_file()
flames.render()
flames.convert_to_movie()
