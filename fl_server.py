
import os

from tensorflow.keras import backend as K
from fl_client import Client
from fl_client import ClientVec

from misc import MetricState
from sklearn.preprocessing import MinMaxScaler
from SPF import TreeModel
from data_preprocess import load_data, get_test_dataset, compute_metrics, test_step

class Server(object):
    def __init__(self, clients, agent_fn, params):
        assert isinstance(clients, ClientVec)
        assert len(clients) > 0

        self.clients = clients
        self.agent = agent_fn()
        self.params = params

        self._train = True

    def send_model(self):
        self.clients.recv_model(self.agent)

    # def aggregate_model(self): # fedavg
    #     avg_weights = self.clients[0].agent.actor.trainable_weights
    #     avg_weights = list(map(lambda x: x.numpy(), avg_weights))
    #     for c in self.clients[1:]:
    #         avg_weights = list(map(lambda x,y: x + y.numpy(), avg_weights, c.agent.actor.trainable_weights))
    #
    #     for i, w in enumerate(self.agent.actor.trainable_weights):
    #         K.set_value(w, avg_weights[i] / len(self.clients))
    #         # layer.set_weights(avg_weights[i] / len(self.clients))

    def aggregate_model(self): # revise
        avg_weights = self.clients[0].agent.actor.trainable_weights
        avg_weights = list(map(lambda x: x.numpy(), avg_weights))
        for c in self.clients[1:]:
            avg_weights = list(map(lambda x,y: x + y.numpy(), avg_weights, c.agent.actor.trainable_weights))

        for i, w in enumerate(self.agent.actor.trainable_weights):
            K.set_value(w, avg_weights[i] / len(self.clients))
            # layer.set_weights(avg_weights[i] / len(self.clients))


    def load(self, path):
        file = f"agent_server.hdf5"
        filepath = os.path.join(path, "checkpoints")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.agent.load_model(os.path.join(filepath, file))

    def save(self, path):
        file = f"agent_server.hdf5"
        filepath = os.path.join(path, "checkpoints")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.agent.save_model(os.path.join(filepath, file))

    def train(self, params):
        self._train = True
        params["out_path"] = os.path.join(params["log_path"], "train")

    def eval(self, params):
        self._train = False
        params["out_path"] = os.path.join(params["log_path"], "test")

    def step(self):
        pass

    def local_test(self, data_path):
        file_list = get_test_dataset(data_path)
        train_X1, train_y1, test_X1, test_y1 = load_data(file_list, seed=self.params["seed"])
        out_path = self.params["out_path"]

        res_state_names = ["precisions", "aucs", "recalls", "f1scores",
                           "fprs", "fnrs"]
        res_states = [MetricState(res_state_names), ]

        precisions_total = []
        for i in range(len(train_X1)):
        # for i in range(2):
            precisions = []
            model1 = TreeModel(window_size=self.params["window_size"],
                               n_trees=self.params["n_trees"],
                               max_depth=self.params["depth"],
                               min_depth=2,
                               terminal_depth=4, adaptive=0.6,
                               seed=10)
            _, structure1 = model1.fit(train_X1[i])
            mms = MinMaxScaler()
            observations1 = mms.fit_transform(structure1)
            step = 0

            res1 = []
            test1 = []
            # f = open(os.path.join(out_path, "action.txt"), 'a')
            # f_ = open(os.path.join(out_path, "node.txt"), 'a')
            for tx1, ty1 in zip(test_X1[i], test_y1[i]):
                test1.append(ty1)
                observations1, res1, actions1, step, node_ids1 = test_step(model1, tx1, ty1, observations1,
                                                                           step,
                                                                           res1, self.agent)

            #     f.write("agent server:{}".format(actions1) + '\n')  # 记录两个agent的action
            #     if len(node_ids1) > 0:
            #         f_.write("agent server:{}".format(node_ids1[0]) + '\n')  # 记录两个agent作用的node的id
            # f.close()

            states = compute_metrics(test1, res1)
            res_states[0].append(res_state_names, states)
            # precisions.append(states[0])
            # precisions_total.append(states[0])

        keys, vals = res_states[0].mean(res_states)
        test_keys = ["test/" + key for key in keys]

        for key, val in zip(keys, vals):
            print("{}:\t{}".format(key, val))

        fp = open(os.path.join(self.params["log_path"], "MARL_DNN.txt"), 'a')
        for i, res in enumerate(res_states):
            fp.write("test\nagent server:\n")
            res_states[i].self_mean()
            for key in res.keys():
                fp.write("{}:{}\n".format(key, res[key]))
            fp.write("\n")
        fp.write("\n")

        fp.close()

        return dict(zip(test_keys, vals))