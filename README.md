# Gym-AvoidShit
This repository contains the Avoid Shit as a gym environment

# Implementation
Works in Ubuntu 18.04  
you need to install X11 in linux system.
# Installation
In gym-avoidshit directory,

you need to type following command

```
This branch run on tensorflow2.2 (also install tensorflow_probability library)
pip3 install -e . (deque, matplot, cpprb, logging)
```

# Train Model
Just needs this command.
python3 Train.py
Also, you can edit Train.py to modify buffer size, step count and so on.
You can change some parameters related to learning directly in SAC.py

# Test Random Action

Edit finalmode.py load_weights code(just uncomment)
In each random_yes_... direcotry has action.h5
Load weights them to test the model.

If you change the test from random to fix.
You should change the gym_avoidshit/envs/avoidshit_env.py

self.RANDOM = True to self.RANDOM = False also opposite case.


```
DISPLAY=:0 python3 score.py
or Just using some other Display Tools like mobaxterm.
```





# Version

## AvoidShit-1.0
Discrete action space , fixed shit generation
## AvoidShit-1.1
Discrete action space , fixed or random shit generation
Continuous action space , fixed or random shit generation


