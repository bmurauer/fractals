from fractals.flame import Flame

flame: Flame = Flame.from_file(file_name="flames.flame", flame_name="019")
flame.xforms[2].add_rotation_animation(2, bpm=120)
flame.xforms[3].add_rotation_animation(3)
flame.xforms[0].add_orbit_animation(radius=0.02, bpm=60)
flame.palette.n_rotations = 1
flames = flame.animate(total_frames=30 * 60 * 3, directory_name="19-beating")
flames.one_file_per_flame = True
flames.render(verbose=True)
flames.convert_to_movie()
