from fractals.flame import Flame, Transform
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("flames/flames.flame", "026", None)
flame = Flame.from_element(xml)
flame.xforms[0].coefs.orbit_transform(radius=0.5)
flame.palette.n_rotations = 1
flames = flame.animate(n_frames=1000)
flames.write_file()
flames.render()
flames.convert_to_movie()
