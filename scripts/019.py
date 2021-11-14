from fractals.flame import Flame, Transform
from fractals.utils import get_flame_from_file

xml = get_flame_from_file("flames/flames.flame", "019")
flame = Flame.from_element(xml)
flame.xforms[2].coefs.fully_rotate_linear(2)
flame.xforms[3].coefs.fully_rotate_linear(3)
flame.xforms[0].coefs.orbit_transform(radius=0.002)
flame.palette.n_rotations = 1
flames = flame.animate(n_frames=1000)
flames.write_file()
flames.render()
flames.convert_to_movie()
