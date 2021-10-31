# Duckietown-Duck-Riders
This repository contains our group project for course VITMAV45.

## <b>Milestone 1</b>
## Running manual_control.py

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
docker pull kajc10/duck-riders
docker run -it duck-riders bash
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
Unique maps can be written/generated and then used with the duckietown-gym simulator.
We prepared a test map and placed it at [/maps](/maps) .

These custom maps have to be copied to the installed packages. 
After placing your map into the [/maps](/maps) folder, you can carry out the copying process by running the `copy_custom_maps.py` script:
```bash
python3 copy_custom_maps.py
```
 It copies all .yaml files from the folder to the destinations. Note that you need to switch folders! <br>Reminder of file structure:
 ```bash
|- gym-duckietown
|   |- manual_control.py
|   |- ...other files ...
|- maps
|   |- rider_map.yaml 
|   |- copy_custom_maps.py
|   |- ...other maps ...

 ```

When starting the manual control script, the map can be selected via the `--map-name` option, e.g.:
```bash
python3 manual_control.py --env-name Duckietown-udem1-v0 --map-name YOURMAPNAME
```
Make sure you pass only the name, the .yaml extension is not needed!
