import numpy as np
import matplotlib.pyplot as plt

class TrainingAnalyzer():
    """
    Framework for visualizing training data logged by TrainEpisodeLogger()
    """
    def __init__(self, filepath):
        try:
            with open(filepath, 'r') as f:
                self.processed_data = list()
                for line in f:
                    line.strip()
                    line.replace("\"",'')
                    data = line.split('\' \'') #cuts out loss metric strings
                    self.processed_data.append([(d.replace('\'','')) for d in data])
                    for value in self.processed_data:
                        if value == []:
                            self.processed_data.remove(value)
        except:
            raise FileNotFoundError()

        self.metrics = {
            'step':list(),
             'nb_steps':list(), 'episode':list(),
             'duration':list(), 'episode_steps':list(),
             'sps':list(), 'episode_reward':list(), 'reward_mean':list(),
              'reward_min':list(),'reward_max':list(),
             'action_mean':list(), 'action_min':list(), 'action_max':list(),
              'obs_mean':list(),
             'obs_min':list(), 'obs_max':list(), 'metrics_text':list()
            }

        for episode in self.processed_data[:-1]:
            if "WALL CLOCK TIME:" in episode[0]:
                continue
            self.metrics['step'].append(float(episode[16]))
            self.metrics['nb_steps'].append(float(episode[8]))
            self.metrics['episode'].append(float(episode[4]))
            self.metrics['duration'].append(float(episode[3]))
            self.metrics['episode_steps'].append(float(episode[6]))
            self.metrics['sps'].append(float(episode[15]))
            self.metrics['episode_reward'].append(float(episode[5]))
            self.metrics['reward_mean'].append(float(episode[13]))
            self.metrics['reward_min'].append(float(episode[14]))
            self.metrics['reward_max'].append(float(episode[12]))
            self.metrics['action_mean'].append(float(episode[1]))
            self.metrics['action_min'].append(float(episode[2]))
            self.metrics['action_max'].append(float(episode[0]))
            self.metrics['obs_mean'].append(float(episode[10]))
            self.metrics['obs_min'].append(float(episode[11]))
            self.metrics['obs_max'].append(float(episode[9]))
            self.metrics['metrics_text'].append(episode[7])

    def graph_metrics_by_episode(self, metric_list=[['episode_reward','-','b']], stylesheet='seaborn', smooth=True):
        """
        Metrics are passed in the form (metric_string_id, linetype, color) according to matplotlib conventions.
        """
        data = list()
        labels = list()
        for metric in metric_list:
            if metric[0] not in self.metrics.keys():
                print(metric[0] + "Not a Valid Metric")
            else:
                data.append(self.metrics[metric[0]]) #appends the raw data
                labels.append(metric[0]) #appends the name of the metric

        plt.style.use('fivethirtyeight')
        plt.xlabel('Episode')
        lines = list()
        for i, metric in enumerate(data):
            metric = self.savitzky_golay(metric, 91, 2)
            if i == 0:
                lines = plt.plot(self.metrics['episode'],metric,label=labels[i])
            else:
                line = (plt.plot(self.metrics['episode'],metric,label=labels[i]))
                lines.append(line[0])

        if len(labels) == 1:
            plt.ylabel(labels[0])
        else:
            plt.legend([line for line in lines],[label for label in labels])


    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        """
        Noise-smoothing function for charts and graphs. From SciPy documentation.
        """
        import numpy as np
        from math import factorial

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( np.array(y[1:half_window+1][::-1]) - np.array(y[0]) )
        lastvals = y[-1] + np.abs( np.array(y[-half_window-1:-1][::-1]) - np.array(y[-1]))
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')

e = TrainingAnalyzer('../model_saves/Vanilla/dqn_MsPacmanDeterministic-v4_REWARD_DATA.txt')
e.graph_metrics_by_episode(metric_list=[['episode_reward','-','b+']])
e = TrainingAnalyzer('../model_saves/PDD/pdd_dqn_MsPacmanDeterministic-v4_REWARD_DATA.txt')
e.graph_metrics_by_episode(metric_list=[['episode_reward','-','b+']])
e = TrainingAnalyzer('../model_saves/NoisyNstepPDD/noisynet_pdd_dqn_MsPacmanDeterministic-v4_REWARD_DATA.txt')
e.graph_metrics_by_episode(metric_list=[['episode_reward','-','b+']])
plt.legend(['Vanilla','PrioritizedDoubleDueling','NoisyNstepPDD'])
plt.show()
