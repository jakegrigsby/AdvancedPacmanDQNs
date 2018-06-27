from analysis_models import noisy_nstep_pdd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import numpy as np

noisy_nstep_pdd.compile(Adam(lr=.00025/4), metrics=['mae'])

weight_names =['noisynet_pdd_dqn_MsPacmanDeterministic-v4_weights_' + str(num) + '.h5f' for num in range(500000,10500000,500000)]

folder_path = '../model_saves/NoisyNstepPDD/'
avg_sigmas = []
for version in weight_names:
    noisy_nstep_pdd.load_weights(folder_path + version)
    avg_sigmas.append(np.mean(noisy_nstep_pdd.layers[-3].get_weights()[1]))

plt.style.use('fivethirtyeight')
plt.xlabel('Timesteps')
plt.ylabel('Average Sigma Weight')
plt.plot([step for step in range(500000,10500000,500000)],avg_sigmas, color='purple',linewidth=3)
plt.show()
