import heapq
import os
import random

import numpy as np
from tensorflow.keras import backend as K
from fl_client import Client
from fl_client import ClientVec

from misc import MetricState
from sklearn.preprocessing import MinMaxScaler
from SPF import TreeModel
from data_preprocess import load_data, get_server_test_dataset, compute_metrics

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

    def aggregate_model_fedavg(self): # fedavg
        id = [i for i in range(0,self.params["client_num"])]
        index = random.sample(id, self.params["client_chosen"])
        avg_weights = self.clients[index[0]].agent.actor.trainable_weights
        avg_weights = list(map(lambda x: x.numpy(), avg_weights))
        for i in index[1:]:
            avg_weights = list(map(lambda x,y: x + y.numpy(), avg_weights, self.clients[i].agent.actor.trainable_weights))

        for i, w in enumerate(self.agent.actor.trainable_weights):
            K.set_value(w, avg_weights[i] / len(self.clients))


    def aggregate_model_confi_random(self):  # confidential index
        id = [i for i in range(0, self.params["client_num"])]
        index = random.sample(id, self.params["client_chosen"])
        self.clients[index[0]].confi_index = self.clients[index[0]].pos_fb / (
                    self.clients[index[0]].pos_fb + self.clients[index[0]].neg_fb)
        weights_actor = self.clients[index[0]].agent.actor.trainable_weights
        weights_critic = self.clients[index[0]].agent.critic.trainable_weights
        weights_actor = list(map(lambda x: x.numpy(), weights_actor))
        weights_critic = list(map(lambda x: x.numpy(), weights_critic))
        weights_actor = (item * self.clients[index[0]].confi_index for item in weights_actor)
        weights_critic = (item * self.clients[index[0]].confi_index for item in weights_critic)
        sum_confi = self.clients[index[0]].confi_index
        for i in index[1:]:
            self.clients[i].confi_index = self.clients[i].pos_fb / (self.clients[i].pos_fb + self.clients[i].neg_fb)
            avg_weights_actors = list(map(lambda x: x.numpy(), self.clients[i].agent.actor.trainable_weights))
            avg_weights_critics = list(map(lambda x: x.numpy(), self.clients[i].agent.critic.trainable_weights))
            avg_weights_actors = (item * self.clients[i].confi_index for item in avg_weights_actors)
            avg_weights_critics = (item * self.clients[i].confi_index for item in avg_weights_critics)
            weights_actor = list(map(lambda x, y: x + y, weights_actor, avg_weights_actors))
            weights_critic = list(map(lambda x, y: x + y, weights_critic, avg_weights_critics))
            sum_confi += self.clients[i].confi_index
        for i, w in enumerate(self.agent.actor.trainable_weights):
            K.set_value(w, weights_actor[i] / sum_confi)
        for i, w in enumerate(self.agent.critic.trainable_weights):
            K.set_value(w, weights_critic[i] / sum_confi)
        for c in self.clients:
            c.pos_fb = 0
            c.neg_fb = 0
            c.confi_index = 0

    def aggregate_model_confi_selec(self):  # confidential index
        confid = []
        for c in self.clients:
            c.confi_index = c.pos_fb/(c.pos_fb+c.neg_fb)
            confid.append(c.confi_index)
        index = heapq.nlargest(self.params['client_chosen'], range(len(confid)), confid.__getitem__)
        self.clients[index[0]].confi_index = self.clients[index[0]].pos_fb / (self.clients[index[0]].pos_fb + self.clients[index[0]].neg_fb)
        weights_actor = self.clients[index[0]].agent.actor.trainable_weights
        weights_critic = self.clients[index[0]].agent.critic.trainable_weights
        weights_actor = list(map(lambda x: x.numpy(), weights_actor))
        weights_critic = list(map(lambda x: x.numpy(), weights_critic))
        weights_actor = (item * self.clients[index[0]].confi_index for item in weights_actor)
        weights_critic = (item * self.clients[index[0]].confi_index for item in weights_critic)
        sum_confi = self.clients[index[0]].confi_index
        for i in index[1:]:
            self.clients[i].confi_index = self.clients[i].pos_fb / (self.clients[i].pos_fb + self.clients[i].neg_fb)
            avg_weights_actors = list(map(lambda x: x.numpy(), self.clients[i].agent.actor.trainable_weights))
            avg_weights_critics = list(map(lambda x: x.numpy(), self.clients[i].agent.critic.trainable_weights))
            avg_weights_actors = (item * self.clients[i].confi_index for item in avg_weights_actors)
            avg_weights_critics = (item * self.clients[i].confi_index for item in avg_weights_critics)
            weights_actor = list(map(lambda x, y: x + y, weights_actor, avg_weights_actors))
            weights_critic = list(map(lambda x, y: x + y, weights_critic, avg_weights_critics))
            sum_confi += self.clients[i].confi_index
        for i, w in enumerate(self.agent.actor.trainable_weights):
            K.set_value(w, weights_actor[i] / sum_confi)
        for i, w in enumerate(self.agent.critic.trainable_weights):
            K.set_value(w, weights_critic[i] / sum_confi)
        for c in self.clients:
            c.pos_fb = 0
            c.neg_fb = 0
            c.confi_index = 0


    def load(self, path):
        file1 = f"agent_server_actor.hdf5"
        file2 = f"agent_server_critic.hdf5"
        filepath = os.path.join(path, "checkpoints")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.agent.load_model(os.path.join(filepath, file1),os.path.join(filepath, file2))

    def save(self, path):
        file1 = f"agent_server_actor.hdf5"
        file2 = f"agent_server_critic.hdf5"
        filepath = os.path.join(path, "checkpoints")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.agent.save_model(os.path.join(filepath, file1), os.path.join(filepath, file2))

    def train(self, params):
        self._train = True
        params["out_path"] = os.path.join(params["log_path"], "train")

    def eval(self, params):
        self._train = False
        params["out_path"] = os.path.join(params["log_path"], "test")

    def step(self):
        pass

    def local_test(self, data_path):
        file_list = get_server_test_dataset(data_path)
        train_X1, train_y1, test_X1, test_y1 = load_data(file_list, seed=self.params["seed"])
        out_path = self.params["out_path"]

        res_state_names = ["precisions", "aucs", "recalls", "f1scores",
                           "fprs", "fnrs"]
        res_states = [MetricState(res_state_names), ]

        confi_index = []

        for i, client in enumerate(self.clients):
            confi_index.append(client.confi_index)

        confi_index_max = confi_index.index(max(confi_index))

        for i in range(len(train_X1)):
        # for i in range(2):
            model1 = self.clients[confi_index_max].tree_model  ## choose the best recognition model as the original model of the new emerging recognition end
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
                observations1, res1, actions1, step, node_ids1 = self.test_step(model1, tx1, ty1, observations1,
                                                                           step, res1)

            #     f.write("agent server:{}".format(actions1) + '\n')  # 记录两个agent的action
            #     if len(node_ids1) > 0:
            #         f_.write("agent server:{}".format(node_ids1[0]) + '\n')  # 记录两个agent作用的node的id
            # f.close()

            states = compute_metrics(test1, res1)
            res_states[0].append(res_state_names, states)
            # precisions.append(states[0])
            # precisions_total.append(states[0])

        keys, vals = res_states[0].mean(res_states)
        test_keys = ["server_test/" + key for key in keys]

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

    # def test_step(self, model, tx, ty, observations, step, res):
    #     mms = MinMaxScaler()
    #     actions = []
    #     node_ids = []
    #     uncertainty = model.predict(tx, return_consistency=True, cut=False)
    #     # if np.mean(np.abs(uncertainty[1])) - 0.5 < 0.3:
    #     if 1:
    #         print('RL test')
    #         for observation in observations:
    #             actions.append(self.agent.choose_action(observation))
    #
    #         _, node_id = model.update_structure(tx, actions, int(ty))
    #         node_ids.append(node_id)
    #
    #         structure_ = []
    #         for atree in model.trees:
    #             structure_.append(model.record_tree(atree))
    #         observations_ = mms.fit_transform(structure_)
    #         predy, pre_list = model.predict(tx, cut=True)
    #         observations = observations_
    #         res.append(predy)
    #     return observations, res, actions, step, node_ids

    def test_step(self, model, tx, ty, observations, step, res):
        mms = MinMaxScaler()
        actions = []
        node_ids = []

        uncertainty = model.predict(tx, return_consistency=True, cut=False)
        # print(uncertainty)
        if np.mean(np.abs(uncertainty[1])) - 0.5 < 0.1:  # propose feedback (can be changed)
        # if 1:
        # if np.random.rand() <0.7:
            print('RL test')
            for observation in observations:
                actions.append(self.agent.choose_action(observation))
            # print("action")
            # print(np.shape(actions))
            _, node_id = model.update_structure(tx, actions, int(ty))
            node_ids.append(node_id)

            structure_ = []
            for atree in model.trees:
                structure_.append(model.record_tree(atree))
            observations_ = mms.fit_transform(structure_)  # normalize the tree structure
            # print(observations_)
            t = 0
            alpha = self.params['alpha']  # reward = alpha * global_reward + (1 - alpha) * regional_reward
            for observation, action, observation_ in zip(observations, actions, observations_):
                predy, pre_list = model.predict(tx, cut=True)

                if predy == int(ty):
                    global_reward = 1
                    # self.pos_fb += 1
                    # print("positive feedback:")
                    # print(RL.pos_fb)
                else:
                    global_reward = -1
                    # self.neg_fb += 1
                    # print("negative feedback:")
                    # print(RL.neg_fb)
                if pre_list[t] == int(ty):
                    regional_reward = 1
                else:
                    regional_reward = -1
                t += 1
                reward = alpha * global_reward + (1 - alpha) * regional_reward

                self.agent.store_transition(observation, action, reward, observation_)
                step += 1
                if step > self.params["step"] and (step % self.params["step"] == 0):
                    self.agent.learn()
                    pass
            observations = observations_
        predy, pre_list = model.predict(tx, cut=True)

        res.append(predy)
        return observations, res, actions, step, node_ids