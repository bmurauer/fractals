from fractals.flame import Flame
from fractals.transitions import pulsating

flame = Flame.from_file("flames.flame", "032", draft=True)
flame.xforms[2].add_attr_animation("ztranslate", 0.17)
flame.xforms[2].add_attr_animation("cylinder", 0.05, envelope=pulsating, bpm=60)
flames = flame.animate(total_frames=240)
flames.render()
flames.convert_to_movie()
