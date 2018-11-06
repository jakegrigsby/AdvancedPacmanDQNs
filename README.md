# AdvancedPacmanDQNs
GitHub repo containing the agents, analysis scripts, and figures from [Advanced DQNs: Playing Pac-man with Deep Reinforcement Learning](https://towardsdatascience.com/advanced-dqns-playing-pac-man-with-deep-reinforcement-learning-3ffbd99e0814)

##### Instructions
###### 1. Download keras-rl
To run agents yourself, you'll need a [keras-rl](https://github.com/keras-rl/keras-rl) version that includes NoisyNetworks, Prioritized Experience Replay, and N-step TD. **For now, that probably means using [this version](https://github.com/jakegrigsby/keras-rl/tree/n-step-dqn). Download it and place keras-rl and AdvancedPacmanDQNs in the same directory.**

###### 2. Download demo observations
If you want to run the CNN visualization scripts, you'll need to download the demo observations (the files are large and annoying to get onto GitHub). You can get them from: [here](https://drive.google.com/open?id=1wfxv1jrzHuguXYQls1jy69bRwKpOkqiY) and [here](https://drive.google.com/open?id=1KYUnZhBVthXvdDCX_2ZjEY3hxwPG1k3O). Place them in the analysis folder.

###### Run Trained Agents
Play trained agents using the commands:
```
python noisyNstepPDD.py --mode test
python pdd.py --mode test
python vanilla.py --mode test
```

Dependencies include: keras, keras-rl, numpy, PIL, cv2, matplotlib, gym.
