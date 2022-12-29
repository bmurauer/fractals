from fractals.flame import Flame

flame = Flame.from_file("flames.flame", "042", draft=True)
flame.add_palette_rotation_animation()
flame.xforms[0].add_orbit_animation(0.1)
flame.xforms[3].add_orbit_animation(0.1)
flame.xforms[3].add_rotation_animation(n_rotations=1)

flames = flame.animate(total_frames=240)

# flames.write_file()
flames.render()
flames.convert_to_movie()
