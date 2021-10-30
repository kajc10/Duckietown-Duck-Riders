# Duckietown-Duck-Riders
This repository contains our group project for course VITMAV45.

## Milestone 1 : run manual_control.py

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

**Opton3: build docker image from Dockerfile**
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
