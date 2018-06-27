"""
Instantiates duplicate (untrained) versions of all of the models found in ../agents, for analysis purposes.
"""
import argparse
import sys
sys.path.append('../../keras-rl') #jakegrigsby/keras-rl fork
from PIL import Image
import numpy as np
import gym
from keras.models import Model
from keras.layers import Flatten, Convolution2D, Input, Dense
from keras.optimizers import Adam
import keras.backend as K
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory, PrioritizedMemory
from rl.core import Processor
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint
from rl.layers import NoisyNetDense

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
env = gym.make('MsPacmanDeterministic-v4')
nb_actions = env.action_space.n

class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory
    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch
    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

input_shape = (WINDOW_LENGTH, INPUT_SHAPE[0], INPUT_SHAPE[1])
frame = Input(shape=(input_shape))
cv1 = Convolution2D(32, kernel_size=(8,8), strides=4, activation='relu', data_format='channels_first')(frame)
cv2 = Convolution2D(64, kernel_size=(4,4), strides=2, activation='relu', data_format='channels_first')(cv1)
cv3 = Convolution2D(64, kernel_size=(3,3), strides=1, activation='relu', data_format='channels_first')(cv2)
dense= Flatten()(cv3)
dense = Dense(512, activation='relu')(dense)
buttons = Dense(nb_actions, activation='linear')(dense)
model = Model(inputs=frame,outputs=buttons)

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()
policy = policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1250000)
vanilla = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, enable_double_dqn=False, enable_dueling_network=False, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)

pr_memory = PrioritizedMemory(limit=1000000, alpha=.6, start_beta=.4, end_beta=1., steps_annealed=10000000, window_length=WINDOW_LENGTH)
pdd = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=pr_memory,
               processor=processor, enable_double_dqn=True, enable_dueling_network=True, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)

input_shape = (WINDOW_LENGTH, INPUT_SHAPE[0], INPUT_SHAPE[1])
frame2 = Input(shape=(input_shape))
cv1 = Convolution2D(32, kernel_size=(8,8), strides=4, activation='relu', data_format='channels_first')(frame)
cv2 = Convolution2D(64, kernel_size=(4,4), strides=2, activation='relu', data_format='channels_first')(cv1)
cv3 = Convolution2D(64, kernel_size=(3,3), strides=1, activation='relu', data_format='channels_first')(cv2)
dense= Flatten()(cv3)
dense = NoisyNetDense(512, activation='relu')(dense)
buttons = NoisyNetDense(nb_actions, activation='linear')(dense)
model = Model(inputs=frame,outputs=buttons)

policy = GreedyQPolicy()
noisy_nstep_pdd = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=pr_memory,
               processor=processor, enable_double_dqn=True, enable_dueling_network=True, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1., n_step=3)
