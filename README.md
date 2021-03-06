# Duckietown-Duck-Riders
```
authors:
Bozsó Katica     - ZE5BJ7  
Pap Bence        - VEJP2J
Pelenczei Bálint - C4QFEY
```

This repository contains our group project for course VITMAV45.  
Some files were copied/derived from the official [Gym-Duckietown repository](https://github.com/duckietown/gym-duckietown)

![Duckieee](./docs/first_try.gif)

## **Project summary:**  
The task was to train autonomous driving agents in simulation, that later can be tested on real hardware as well. We used the Gym-Duckietown Simulator - which is built on top of Open AI Gym - provided by the official Duckietown project.


## Training
For the training we decided to use the [Ray](https://docs.ray.io/en/latest/) framework.  
>**Ray** provides a simple, universal API for building distributed applications.

It is packaged with other libraries for accelerating machine learning workloads, of which we used [RLlib](https://docs.ray.io/en/latest/rllib.html) and [Tune](https://docs.ray.io/en/master/tune/index.html).

>**RLlib** is an industry-grade library for reinforcement learning (RL). It offers both high scalability and a unified API for a variety of applications.

>**Tune** is a Python library for experiment execution and hyperparameter tuning at any scale.

RLlib comes with built-in NN models (that can be customized), but an algorithm for the training had to be chosen.  
Based on comparisions, we opted for **Proximal Policy Optimization** [(PPO)](https://docs.ray.io/en/latest/rllib-algorithms.html#ppo).

## Testing
The trained model can be easily tested with the help of the Simulator. Given any observation (image), the trained agent can compute the best `action` from the action space. In our case we need 2 values - velocity of left and right motors. Then the environment's `step` function can be called with the computed action passed as an argument. As the action is executed, the environment returns the next observation and this process continues.. The env's `render` function makes the self-driving agent visible.

## Wrappers
Wrappers are quite important, since they allow us  to add functionality to environments - modify rewards, observations etc.
We used the original gym_duckietown wrappers with slight modifications for basic observation processing, like image resizing or normalizing.
In addition, we prepared 2 custom wrappers:
- CropWrapper - crops the sky from images
- DtRewardWrapper - feeds agent with custom reward, based on several aspects like speed, heading angle and position in line

## Files
- `src/`  - contains the simulator
- `train.py` - training setup
- `trainers.py` - different training agents and their hyperparameter setup
- `test.py`  - for testing the trained agent
- `manual_control.py` - manually driving (modified for plotting reward)
- `wrappers.py` - to modify env's observation and rewards
- `setup.sh`  - for installing gym dependencies
- `dump/` - checkpoints are dumped here
- `maps/` - custom maps are placed here

## Instructions
There are 2 options available for testing our repository.

### **Option1: clone this repository**  
Issue the following commands:
```bash
git clone https://github.com/kajc10/Duckietown-Duck-Riders
cd Duckietown-Duck-Riders
pip3 install -e . #install dependencies
```  

Run training with:
- `train.py` 
and specify which algorithm to use with:
- `--mode [algorithm]` - `[algorithm]` can either be `ddpg`, `ppo` or `a2c` (default is ppo)


:warning:Make sure you use the following options correctly unless the training won't work!
- `--num_cpus` - according to your system
- `--num_gpus` - according to your system  

Other less important arguments:
- `--seed` - place random seed (default is 10)
- `--map_name` -  mapname without .yaml extension (default is small_loop)
- `--max_steps` - until the simulation stops (default is 2000000)
- `--training_name` - output folder (default is Training_results)

Example for full command:
```bash
python3 train.py --mode ppo --num_cpus 3 --num_gpus 0 --seed 10 --map_name small_loop --max_steps 2000000 --training_name Training_results
```


Run test:  
You can test your trained models or others' if you place checkpoint file in a folder to [./dump/Training_results](/dump/Training_r_esults). E.g. */dump/Training_results/best_model/checkpoint_000245*  
The folder of the model should be given as argument.
Also a map name can be passed.
```bash
python3 test.py --map_name small_loop --checkpoint_folder best_model --mode ppo
```

<br>

### **Option2: pull our docker image from Docker-Hub**  
Most of the necessary dependencies are installed in this image,
but to run training on GPUs, you will need to install cuda.
```bash
docker pull kajc10/duck-riders:milestone3
docker run -it kajc10/duck-riders:milestone3 bash
```
Due to docker, a virtual display has to be created:
```bash
Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log & export DISPLAY=:0
```

For the case you want to build a new image, a Dockerfile is provided!

Run the commands specified in **option 1**.
Note that you might start your command with `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1` . Also, if you are using the image through ssh, make sure you use `nohup`, so in case of network error the script won't terminate.

## Additional features
### **Loading custom maps**  

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
|   |- ETHZ_autolab_technical_track.yaml
|   |- copy_custom_maps.py
|   |- ... other maps ...
| other_folders ...

 ```

When starting the manual control script, the map can be selected via the `--map-name` option, e.g.:
```bash
python3 manual_control.py --map-name rider_map
```
Make sure you pass only the name, the .yaml extension is not needed!

Note: For the map generation we used the [duckietown/map-utils](https://github.com/duckietown/map-utils) repository. The `tile_size` had to be added to the last row manually. We are aware of the possibility of writing .yaml files manually, but for our current needs we favourized the generator.

Video of our running custom map: https://youtu.be/sgJBtslqAe0

<br>

### **Modified manual_control.py**   
To test our reward wrapper, we modified the original `manual_control.py` so that it always shows the current reward thus we are able to check the actual rewards during a manual testing.
