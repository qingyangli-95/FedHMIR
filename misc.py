
import os
import dowel
import random
import numpy as np

from collections import OrderedDict
from dowel import TabularInput

def make_dirs(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def get_log_path(par_path, exp_name):
    exp_name = "exp-" + exp_name + "-"
    log_path = None
    exps = os.listdir(par_path)
    for i in range(100):
        log_folder = "{}-{:03d}".format(exp_name, i)
        log_path = os.path.join(par_path, log_folder)
        if log_folder not in exps:
            make_dirs([log_path])
            break
    return log_path

class MetricState:
    def __init__(self, state_name_list):
        self.states = OrderedDict()
        for name in state_name_list:
            self.states[name] = []

    def __getitem__(self, key):
        return self.states[key]

    def keys(self):
        return self.states.keys()

    def __len__(self):
        return len(self.states)

    def append(self, key_list, state_list):
        for i, key in enumerate(key_list):
            self.states[key].append(state_list[i])

    def _mean_key(self, key, metric_list):
        val_list = []
        for metric in metric_list:
            val_list.extend(metric[key])
        return np.mean(val_list)

    def mean(self, metric_list):
        if len(metric_list) == 0:
            raise ValueError("state_list is empty")
        key_list = metric_list[0].keys()
        metric_list = [metric_list for _ in range(len(metric_list[0]))]
        res = list(map(self._mean_key, key_list, metric_list))
        return key_list, res

    def self_mean(self):
        for key, val in self.states.items():
            self.states[key] = [np.mean(val)]

    def __repr__(self):
        out = "{"
        for key, val in self.states.items():
            out += str(key) + ": " + str(val) + ", "
        out += "}"
        return out

if __name__ == '__main__':
    name_list = ["a", "b"]
    m = MetricState(name_list)
    m.append(name_list, [[0.1], [0.2]])
    m.append(name_list, [[0.5], [0.3]])
    m.append(name_list, [[0.4], [1]])
    k, r = m.mean([m])
    print(k)
    print(r)

    m.self_mean()
    print(m)


class Logger:
    def __init__(self, log_path, batch_or_epoch="batch"):
        self.batch_or_epoch = batch_or_epoch
        self.logger = dowel.logger
        self.tabular = dowel.tabular
        self.logger.add_output(dowel.StdOutput())
        self.logger.add_output(dowel.CsvOutput(os.path.join(log_path, "progress.csv")))
        self.logger.add_output(dowel.TextOutput(os.path.join(log_path, "debug.log")))

    def log(self, data):
        self.logger.log(data)

    def log_agent(self, agent_idx, key, val):
        self.logger.log("agent {} |{}: {}".format(agent_idx, key, val))

    def record(self, states):
        tabular = TabularInput()
        for key in states.keys():
            tabular.record(key, states[key])
        self.logger.log(tabular)

    def dump(self):
        self.logger.dump_all()


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)