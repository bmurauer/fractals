import math
from copy import deepcopy

from fractals.flame import Flame
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("flames.flame", "039")
flame = Flame.from_element(xml)
flame.add_palette_rotation_animation()
flame.xforms[0].add_scale_animation(0.8, n_repetitions=3)
flame.xforms[1].add_scale_animation(0.8, n_repetitions=2)

transform_target = deepcopy(flame.xforms[2].fransform)
transform_target.rotate(radiants=(3 * math.pi / 180))

flame.xforms[2].add_translation_animation(transform_target)
flame.xforms[1].add_scale_animation(0.95)
flames = flame.animate(total_frames=1500)
# flames.write_file()
flames.render()
flames.convert_to_movie()
