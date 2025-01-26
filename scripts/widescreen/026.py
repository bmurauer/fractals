from fractals.flame import Flame

flame = Flame.from_file("flames.flame", "026", draft=True)
flame.xforms[0].add_orbit_animation(radius=0.5, bpm=40, beat_bumpyness=0.75)
flame.xforms[1].add_orbit_animation(radius=0.1, bpm=120)
flame.palette.n_rotations = 1
flames = flame.animate(total_frames=900, directory_name="26-beating")
flames.render()
flames.convert_to_movie()
