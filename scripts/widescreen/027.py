from fractals.flame import Flame

flame: Flame = Flame.from_file("flames.flame", "027", draft=True)
flame.add_palette_rotation_animation()
flame.xforms[0].add_rotation_animation(bpm=60)
flame.xforms[0].add_scale_animation(factor=1.5, n_repetitions=2, bpm=120)
flame.xforms[2].add_rotation_animation(n_rotations=2)
flame.xforms[2].add_attr_animation("julian", 1.2, bpm=90)
flames = flame.animate(total_frames=600, directory_name="27-beating")
flames.render()
flames.convert_to_movie()
