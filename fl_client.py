import os
import numpy as np
from misc import MetricState
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from SPF import TreeModel
from data_preprocess import load_data, compute_metrics, rl_step

class ClientVec(object):
    def __init__(self, client_num, datasets, agent_fn, params):
        self.clients = [Client(i, datasets[i], agent_fn, params)
                        for i in range(client_num)]
        self.client_num = client_num

    def recv_model(self, model):
        for c in self.clients:
            c.recv_model(model)

    def get_train_steps(self):
        return max([len(c) for c in self.clients])

    def __len__(self):
        return self.client_num

    def __getitem__(self, item):
        return self.clients[item]

    def train_step(self):
        cli_states = []
        for c in self.clients:
            cli_states.append(c.train_step())
        return self.convert_states(cli_states)

    def convert_states(self, cli_states):
        output_states = {}
        if len(self.clients) > 0:
            states = np.stack(cli_states)
            for j, key in enumerate(self.clients[0].res_state_names):
                output_states[key] = states[:, j].mean()
        return output_states


    def log_summary(self, params):
        res_states = [c.res_state for c in self.clients]
        keys, vals = res_states[0].mean(res_states)

        for key, val in zip(keys, vals):
            print("{}:\t{}".format(key, val))

        keys, vals = res_states[0].mean(res_states)
        train_keys = ["train/" + key for key in keys]

        fp = open(os.path.join(params["log_path"], "MARL_DNN.txt"), 'a')
        fp.write("window_size: " + str(params["window_size"]) + " trees: " + str(params["n_trees"]) +
                 " depth:" + str(params["depth"]) + '\n')
        fp.write("params:" + str(params) + '\n')

        for key, val in zip(keys, vals):
            fp.write("{}:{}\n".format(key, val))
        fp.write("\n")

        for i, res in enumerate(res_states):
            fp.write("agent{}:\n".format(i))
            res_states[i].self_mean()
            for key in res.keys():
                fp.write("{}:{}\n".format(key, res[key]))
            fp.write("\n")
        fp.write("\n")

        fp.close()

        return dict(zip(train_keys, vals))

    def save(self, path):
        for c in self.clients:
            c.save(path)

    def load(self, path):
        for c in self.clients:
            c.load(path)

    def send_updated_model(self, server):
        server.aggregate_model(self)

class Client(object): # one edge node and multiple recognition ends
    def __init__(self, id, dataset, agent_fn, params):
        self._load_data(dataset)
        self.id = id
        self.agent = agent_fn()
        self.params = params
        self.cnt = 0

        self.res_state_names = ["precisions", "aucs", "recalls", "f1scores",
                           "fprs", "fnrs"]
        self.res_state = MetricState(self.res_state_names)

    def _load_data(self, dataset_dir):
        self.train_X, self.train_y, self.test_X, self.test_y\
            = load_data(dataset_dir, seed=1)

    def __len__(self):
        return len(self.train_X)

    def _check_iterate(self):
        if self.cnt >= len(self.train_X):
            self.cnt = 0

    def train_step(self):
        out_path = self.params["out_path"]
        model1 = TreeModel(window_size=self.params["window_size"],
                           n_trees=self.params["n_trees"],
                           max_depth=self.params["depth"],
                           min_depth=2,
                           terminal_depth=4, adaptive=0.6,
                           seed=10)

        self._check_iterate()
        _, structure1 = model1.fit(self.train_X[self.cnt])  # 初步树模型得到: dimension: tree_num * (node_num*info_num)

        mms = MinMaxScaler()  # try to normalize the tree structure
        observations1 = mms.fit_transform(structure1)
        step = 0

        res1 = []
        test1 = []
        # f = open(os.path.join(out_path, "action.txt"), 'a')
        # f_ = open(os.path.join(out_path, "node.txt"), 'a')
        for tx1, ty1 in zip(self.test_X[self.cnt], self.test_y[self.cnt]):
            test1.append(ty1)
            observations1, res1, actions1, step, node_ids1 = rl_step(model1, tx1, ty1, observations1, step,
                                                                     res1, self.agent)  # agent1
            # observations1, res1, actions1, step, node_ids1 = test_step(model1, tx1, ty1, observations1, step,
            #                                                          res1, self.agent)  # agent1
        #     f.write("agent{}:{}".format(self.id + 1, actions1) + '\n')  # 记录两个agent的action
        #     if len(node_ids1) > 0:
        #         f_.write("agent{}:{}".format(self.id + 1, node_ids1[0]) + '\n')  # 记录两个agent作用的node的id
        # f.close()
        # f_.close()

        states = compute_metrics(test1, res1)
        self.res_state.append(self.res_state_names, states)

        return states

    def load(self, path):
        file = f"client_{self.id}.hdf5"
        filepath = os.path.join(path, "checkpoints")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.agent.load_model(os.path.join(filepath, file))


    def save(self, path):
        file = f"client_{self.id}.hdf5"
        filepath = os.path.join(path, "checkpoints")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.agent.save_model(os.path.join(filepath, file))


    def recv_model(self, model):
        for w, model_w in zip(self.agent.actor.trainable_weights,
                                      model.actor.trainable_weights):
            K.set_value(w, model_w.numpy())
            # layer.set_weights(model_layer.get_weights())

    def send_updated_model(self, server):
        pass