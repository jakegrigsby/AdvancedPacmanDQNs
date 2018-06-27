from analysis_models import noisy_nstep_pdd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
from PIL import Image
import cv2

model = noisy_nstep_pdd
model.compile(Adam(lr=.00025/4), metrics=['mae'])
model.load_weights('../model_saves/NoisyNstepPDD/final_noisynet_nstep_pdd_dqn_MsPacmanDeterministic-v4_weights_40000000.h5f')


def neuron_activations(obs, channel1, channel2, channel3, channel4):
    """
    Visualizing first-layer activations, superimposed over the most recent frame
    of the state's stack. Compares pre-processed screenshot to the activations of
    four chosen channels.
    """
    global model
    assert obs > 4 and obs < 1788, "Chosen observation is outside the range of the demonstration episode."
    for chan in [channel1, channel2, channel3, channel4]:
        assert chan <= 31, "There are only 32 channels available (max index 31)."
    saved_observations = np.load('saved_observations.npy')
    conv_net_outputs = [model.model.layers[1].output]
    conv_only_model = Model(inputs=model.model.input, outputs=conv_net_outputs)
    state = saved_observations[obs-3:obs+1]
    state_batch = np.expand_dims(state, axis=0)
    activations = conv_only_model.predict(state_batch)
    frame = Image.fromarray(state[-1])
    bg = np.array(frame.resize((20,20)))

    _, arr = plt.subplots(1,5)
    arr[0].set_title('Frame Input')
    arr[0].imshow(frame, cmap='gray')
    arr[1].set_title('Channel {}'.format(channel1))
    arr[1].imshow(bg + activations[0,channel1,:,:] * 12, cmap='gray')
    arr[2].set_title('Channel {}'.format(channel2))
    arr[2].imshow(bg + activations[0,channel2,:,:] * 12, cmap='gray')
    arr[3].set_title('Channel {}'.format(channel3))
    arr[3].imshow(bg + activations[0,channel3,:,:] * 12, cmap='gray')
    arr[4].set_title('Channel {}'.format(channel4))
    arr[4].imshow(bg + activations[0,channel4,:,:] * 12, cmap='gray')
    plt.show()

#neuron_activations(315, 13, 16, 22, 1) (ghosts)
#neuron_activations(900, 11, 28, 29, 0) (other)
#neuron_activations(777, 21, 23, 5, 7) (noise)

def decision_heatmaps(obs):
    """
    A channels-first variant of the class activation map algorithm, with heatmaps
    superimposed over the most recent frame of the state's stack.
    """
    global model
    assert obs > 4 and obs < 1788, "Chosen observation is outside the range of the demonstration episode."
    saved_observations = np.load('saved_observations.npy')
    state = saved_observations[obs-3:obs+1]
    state_batch = np.expand_dims(state, axis=0)
    q_vals = model.compute_q_values(state)
    print(q_vals)
    decision = np.argmax(q_vals)
    print(decision)
    decision_encodings = ['None','Up','Right','Left','Down','Right-Up','Left-Up','Right-Down','Left-Down']
    decision_node = model.model.output[:, decision]
    last_conv_layer = model.model.layers[3]
    grads = K.gradients(decision_node, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0,2,3))
    frame = Image.fromarray(state[-1])
    iterate = K.function([model.model.input],[pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([state_batch])
    for i in range(64):
        conv_layer_output_value[i, :, :] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (84, 84))
    heatmap = np.uint8(255 * heatmap)
    superimposed_img = heatmap * .3 + np.array(frame) * .35

    plt.xlabel(decision_encodings[decision])
    plt.imshow(superimposed_img, cmap='gray')
    plt.show()

#decision_heatmaps(490)
#decision_heatmaps(801)

def show_processing(obs_num):
    """
    Compares raw frame input to downsized and grayscaled version that agent sees.
    """
    processed_observations = np.load('saved_observations.npy')
    raw_observations = np.load('raw_observations.npy')
    assert len(processed_observations) == len(raw_observations)
    assert obs_num < len(raw_observations), "Observation index out of episode bounds."
    raw_frame = Image.fromarray(raw_observations[obs_num])
    _, arr = plt.subplots(1,2)
    arr[0].set_title('Raw Input')
    arr[0].imshow(raw_frame)
    arr[1].set_title('Processed Observation')
    arr[1].imshow(processed_observations[obs_num], cmap='gray')
    plt.show()

#show_processing(2)

def show_obs_stack(obs_num):
    """
    Visualizes the frame stack that makes up the model input.
    """
    processed_observations = np.load('saved_observations.npy')
    assert obs_num < len(processed_observations) and obs_num >= 3, "Observation index out of episode bounds."
    _, arr = plt.subplots(1,4)
    arr[0].set_title('t-3')
    arr[0].imshow(processed_observations[obs_num-3], cmap='gray')
    arr[1].set_title('t-2')
    arr[1].imshow(processed_observations[obs_num-2], cmap='gray')
    arr[2].set_title('t-1')
    arr[2].imshow(processed_observations[obs_num-1], cmap='gray')
    arr[3].set_title('t')
    arr[3].imshow(processed_observations[obs_num], cmap='gray')
    plt.show()

#show_obs_stack(200)
