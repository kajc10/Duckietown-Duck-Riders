import gym_duckietown
import duckietown_world
import os
from pathlib import Path

def copy_custom_maps(src, dst):
    assert src.exists(), "Source dir not found at {}".format(src)
    assert dst.exists(), "Destination dir not found at {}".format(dst)
    print("\nCopying maps to {}".format(dst))
    os.system('cp -arv {}/*.yaml {}/'.format(src, dst))

mymaps_path = Path(__file__).resolve().parent  #maps folder
gym_path = Path(gym_duckietown.__file__).parent
duckietownworld_path = Path(duckietown_world.__file__).parent

gym_maps_path = gym_path.joinpath('maps')
duckietownworld_maps_path = duckietownworld_path.joinpath('data', 'gd1', 'maps')

copy_custom_maps(mymaps_path, gym_maps_path)
copy_custom_maps(mymaps_path, duckietownworld_maps_path)

