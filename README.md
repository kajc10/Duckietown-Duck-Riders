# Duckietown-Duck-Riders
This repository contains our group project for course VITMAV45.  
**Tasks:**
- train autonomous driving agents in simulation
- test them on real hardware

Some files were copied/derived from the official Gym-Duckietown repository: https://github.com/duckietown/gym-duckietown  
More info about the official Duckietown project: https://www.duckietown.org/  
Documentations: https://docs.duckietown.org/daffy/


## <b>Milestone 2</b> 

## Initial instructions
For testing, clone this repository and install the neccessary dependencies by issuing the following commands:
```bash
git clone https://github.com/kajc10/Duckietown-Duck-Riders
cd Duckietown-Duck-Riders
pip3 install -e .
```

## Training our model (not optimized yet)
For the training we decided to use the [Ray](https://docs.ray.io/en/latest/) framework.
>**Ray** provides a simple, universal API for building distributed applications.

It is packaged with other libraries for accelerating machine learning workloads, of which we used [RLlib](https://docs.ray.io/en/latest/rllib.html) and [Tune](https://docs.ray.io/en/master/tune/index.html).

>**RLlib** is an industry-grade library for reinforcement learning (RL), built on top of Ray. It offers both high scalability and a unified API for a variety of applications.

>**Tune** is a Python library for experiment execution and hyperparameter tuning at any scale.


List of available algorithms: https://docs.ray.io/en/latest/rllib-toc.html#algorithms

Other helpful links: https://www.anyscale.com/blog/an-introduction-to-reinforcement-learning-with-openai-gym-rllib-and-google


For Milestone 2, an RLlib built-in model was used. It can be modified via config, or a custom one can be created as well.
See more at:  https://docs.ray.io/en/latest/rllib-models.html

Before running, extra dependencies need to be installed:
```bash
pip install -U ray[tune]   # installs Ray + dependencies for Ray Tune
pip install -U ray[rllib]  # installs Ray + dependencies for Ray RLlib
```

Commands:
```bash
cd Duckietown-Duck-Riders

#Run train
python3 train_PPO.py

#Run manual control
python3 manual_control.py --env-name Duckietown-udem1-v0
```
new files:
- `wrappers.py`
- `train_PPO.py`
- `test.py` - during training there was still a slight problem with a wrapper, so instead of using calculated action, only steps with a sample.
Also checkpoint file was hardcoded...

## <del> <b>Milestone 1</b>

## <del>Running manual_control.py
:warning: Warning!!
Folder structure has been reorganized for MS2. The following commands are no longer working for the current version as described. New documentation is coming soon...

**Option1: clone this repository and run manually**
<br>Note that several dependencies need to be installed on your system:<br> [Python 3.6+, OpenAI gym, NumPy, Pyglet, PyYAML,cloudpickle, PyTorch]

Commands:
```bash
git clone https://github.com/kajc10/Duckietown-Duck-Riders
cd Duckietown-Duck-Riders/gym-duckietown
pip3 install -e . #to install dependencies
python3 manual_control.py --env-name Duckietown-udem1-v0
```
<br>

**Option2: pull our docker image from Docker-Hub**
<br>The necessary dependencies are installed in this image.
```bash
docker pull kajc10/duck-riders:v1.0
docker run -it kajc10/duck-riders:v1.0 bash
```
Due to docker, a virtual display has to be created:
```bash
Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log & export DISPLAY=:0
```
Run manual_control.py:
```bash
python3 manual_control.py --env-name Duckietown-udem1-v0
```
<br>

**Option3: build docker image from Dockerfile**
<br>Commands:
```bash
git clone https://github.com/kajc10/Duckietown-Duck-Riders
cd Duckietown-Duck-Riders
docker build -t duck-riders .
```
To run the image:
```bash
docker run -it duck-riders bash
```
To run manual control (as described before) :
```bash
Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log & export DISPLAY=:0
python3 manual_control.py --env-name Duckietown-udem1-v0
```
<br>


## Loading custom maps
:green_heart: Updated for MS2.

Unique maps can be written/generated and then used with the duckietown-gym simulator.
We prepared a test map and placed it at [/maps](/maps) .

These custom maps have to be copied to the installed packages. 
After placing your map into the [/maps](/maps) folder, you can carry out the copying process by running the `copy_custom_maps.py` script:
```bash
python3 copy_custom_maps.py
```
 It copies all .yaml files from the folder to the destinations. Note that you need to switch folders! <br>Reminder of file structure:
 ```bash
|-  .
|   |- manual_control.py
|   |- train_PPO.py
|   |- setup.py
|   |- ... other files ...
|- maps
|   |- rider_map.yaml 
|   |- copy_custom_maps.py
|   |- ... other maps ...
| other_folders ...

 ```

When starting the manual control script, the map can be selected via the `--map-name` option, e.g.:
```bash
python3 manual_control.py --env-name Duckietown-udem1-v0 --map-name rider_map
```
Make sure you pass only the name, the .yaml extension is not needed!

Note: For the map generation we used the [duckietown/map-utils](https://github.com/duckietown/map-utils) repository. The `tile_size` had to be added to the last row manually. We are aware of the possibility of writing .yaml files manually, but for the current milestone we favourized the generator.

Video of our running custom map: https://youtu.be/sgJBtslqAe0
