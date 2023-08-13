import os
import numpy as np
from misc import MetricState
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from data_preprocess import load_data, compute_metrics
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


class ClientVec(object):
    def __init__(self, client_num, datasets, agent_fn, params, pos_fb, neg_fb, tree):
        self.clients = [Client(i, datasets[i], agent_fn, params, pos_fb, neg_fb, tree)
                        for i in range(client_num)]
        self.client_num = client_num

    def recv_model(self, model):
        for c in self.clients:
            c.recv_model(model)

    def get_train_steps(self):
        # print("length of the client:")
        # print(len(self.clients[0]))
        return max([len(c) for c in self.clients])

    def get_test_steps(self):
        # print("length of the client:")
        # print(len(self.clients[0]))
        return max([len(c) for c in self.clients])

    def __len__(self):
        return self.client_num

    def __getitem__(self, item):
        return self.clients[item]

    def train_step(self):
        cli_states = []
        for c in self.clients:
            states = c.train_step()
            cli_states.append(states)
        return self.convert_states(cli_states)

    def test_step(self):
        cli_states = []
        for c in self.clients:
            states = c.test_step()
            cli_states.append(states)
        return self.convert_states(cli_states)

    def convert_states(self, cli_states):
        output_states = {}
        if len(self.clients) > 0:
            states = np.stack(cli_states)
            for j, key in enumerate(self.clients[0].res_state_names):
                output_states[key] = states[:, j].mean()
        return output_states


    def train_log_summary(self, params):
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

    def test_log_summary(self, params):
        res_states = [c.res_state for c in self.clients]
        keys, vals = res_states[0].mean(res_states)

        for key, val in zip(keys, vals):
            print("{}:\t{}".format(key, val))

        keys, vals = res_states[0].mean(res_states)
        train_keys = ["test/" + key for key in keys]

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

    def plot_loss(self):
        plt.figure(figsize=(9, 5))
        for c in self.clients:
            x = np.arange(len(c.agent.critic_loss))
            y = np.array(list(reversed(c.agent.critic_loss))).flatten()
            f = np.polyfit(x,y,20)
            p1 = np.poly1d(f)
            plt.plot(x, p1(x), linewidth=3, label="Client " + str(c.id))
        font = font_manager.FontProperties(family='Times New Roman', size=28)
        plt.legend(loc='upper right', prop=font, ncol=1, fancybox=True, shadow=False)
        plt.xticks(fontsize=28, fontname='Times New Roman')
        plt.yticks(fontsize=28, fontname='Times New Roman')
        plt.xlabel('Training Step', fontsize=28, fontname = 'Times New Roman')
        plt.ylabel('Critic Loss', fontsize=28, fontname = 'Times New Roman')
        plt.savefig("critic convergence.pdf", bbox_inches='tight')
        # plt.show()

        plt.figure(figsize=(9, 5))
        for c in self.clients:
            x = np.arange(len(c.agent.actor_loss))
            y = np.array(list(reversed(c.agent.actor_loss))).flatten()
            f = np.polyfit(x, y, 20)
            p1 = np.poly1d(f)
            plt.plot(x, p1(x), linewidth=3, label="Client " + str(c.id))
        font = font_manager.FontProperties(family='Times New Roman', size=28)
        plt.legend(loc='lower right', prop=font, ncol=1, fancybox=True, shadow=False)
        plt.xticks(fontsize=28, fontname='Times New Roman')
        plt.yticks(fontsize=28, fontname='Times New Roman')
        plt.xlabel('Training Step', fontsize=28, fontname='Times New Roman')
        plt.ylabel('Actor Loss', fontsize=28, fontname='Times New Roman')
        plt.savefig("actor convergence.pdf", bbox_inches='tight')
        # plt.show()

        plt.figure(figsize=(9, 5))
        for c in self.clients:
            plt.plot(np.arange(len(c.agent.reward)), c.agent.reward, linewidth=3, label="Client " + str(c.id))
        font = font_manager.FontProperties(family='Times New Roman', size=28)
        plt.legend(loc='lower right', prop=font, ncol=1, fancybox=True, shadow=False)
        plt.xticks(fontsize=28, fontname='Times New Roman')
        plt.yticks(fontsize=28, fontname='Times New Roman')
        plt.xlabel('Training Step', fontsize=28, fontname='Times New Roman')
        plt.ylabel('Reward', fontsize=28, fontname='Times New Roman')
        plt.savefig("reward.pdf", bbox_inches='tight')
        # plt.show()


class Client(object):
    def __init__(self, id, dataset, agent_fn, params, pos_fb, neg_fb, tree):
        self._load_data(dataset)
        self._load__test_data(dataset)
        self.id = id
        self.agent = agent_fn()
        self.params = params
        self.cnt = 0
        self.pos_fb = pos_fb  # record positive feedback of the client
        self.neg_fb = neg_fb  # record negative feedback of the client
        self.confi_index = 0
        self.tree_model = tree

        self.res_state_names = ["precisions", "aucs", "recalls", "f1scores",
                           "fprs", "fnrs"]
        self.res_state = MetricState(self.res_state_names)

    def _load_data(self, dataset_dir):
        self.train_X, self.train_y, self.test_X, self.test_y\
            = load_data(dataset_dir, seed=1)

    def _load__test_data(self, dataset_dir):
        self.train_X1, self.train_y1, self.test_X1, self.test_y1 \
            = load_data(dataset_dir, seed=2)

    def __len__(self):
        return len(self.train_X)

    def _check_iterate(self):
        self.cnt += 1
        if self.cnt >= len(self.train_X):
            self.cnt = 0

    def train_step(self):
        out_path = self.params["out_path"]
        self._check_iterate()
        _, structure1 = self.tree_model.fit(self.train_X[self.cnt])  # 初步树模型得到: dimension: tree_num * (node_num*info_num)

        mms = MinMaxScaler()  # try to normalize the tree structure
        observations1 = mms.fit_transform(structure1)
        step = 0

        res1 = []
        test1 = []
        # f = open(os.path.join(out_path, "action.txt"), 'a')
        # f_ = open(os.path.join(out_path, "node.txt"), 'a')
        for tx1, ty1 in zip(self.test_X[self.cnt], self.test_y[self.cnt]):
            test1.append(ty1)
            # observations1, res1, actions1, step, node_ids1 = rl_step(self.tree_model, tx1, ty1, observations1, step,
            #                                                          res1, self.agent, self.pos_fb, self.neg_fb)
            observations1, res1, actions1, step, node_ids1 = self.rl_step_train(tx1, ty1, observations1, step, res1)

        states = compute_metrics(test1, res1)
        self.res_state.append(self.res_state_names, states)

        return states

    def load(self, path):
        file1 = f"client_{self.id}_actor.hdf5"
        file2 = f"client_{self.id}_critic.hdf5"
        filepath = os.path.join(path, "checkpoints")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.agent.load_model(os.path.join(filepath, file1),os.path.join(filepath, file2))


    def save(self, path):
        file1 = f"client_{self.id}_actor.hdf5"
        file2 = f"client_{self.id}_critic.hdf5"
        filepath = os.path.join(path, "checkpoints")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.agent.save_model(os.path.join(filepath, file1), os.path.join(filepath, file2))


    def recv_model(self, model):  # ablation of local model and global model
        b = self.params["b"]
        for w, model_w in zip(self.agent.actor.trainable_weights,
                                      model.actor.trainable_weights):
            # K.set_value(w, w * b + model_w.numpy() * (1-b))
            K.set_value(w, w * b + model_w.numpy() * (1 - b))
            # layer.set_weights(model_layer.get_weights())
        for w, model_w in zip(self.agent.critic.trainable_weights,
                                      model.critic.trainable_weights):
            K.set_value(w, w * b + model_w.numpy() * (1 - b))


    def send_updated_model(self, server):
        pass


    def rl_step_train(self, tx, ty, observations, step, res):
        mms = MinMaxScaler()
        actions = []
        node_ids = []

        uncertainty = self.tree_model.predict(tx, return_consistency=True, cut=False)
        # print(uncertainty)
        if np.mean(np.abs(uncertainty[1])) - 0.5 < 0.1:  # propose feedback (can be changed)
        # if 1:
            # if np.random.rand() <0.7:
            print('RL')
            for observation in observations:
                actions.append(self.agent.choose_action(observation))
            # print("action")
            # print(np.shape(actions))
            _, node_id = self.tree_model.update_structure(tx, actions, int(ty))
            node_ids.append(node_id)

            structure_ = []
            for atree in self.tree_model.trees:
                structure_.append(self.tree_model.record_tree(atree))
            observations_ = mms.fit_transform(structure_)  # normalize the tree structure
            # print(observations_)
            t = 0
            alpha = self.params['alpha']  # reward = alpha * global_reward + (1 - alpha) * regional_reward
            for observation, action, observation_ in zip(observations, actions, observations_):
                predy, pre_list = self.tree_model.predict(tx, cut=True)

                if predy == int(ty):
                    global_reward = 1
                    self.pos_fb += 1
                    # print("positive feedback:")
                    # print(RL.pos_fb)
                else:
                    global_reward = -1
                    self.neg_fb += 1
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
                    # not learn model, test what else performance difference
                    self.agent.learn()
                    pass
            observations = observations_
        predy, pre_list = self.tree_model.predict(tx, cut=True)

        res.append(predy)
        return observations, res, actions, step, node_ids


    def test_step(self):
        self._check_iterate()
        _, structure1 = self.tree_model.fit(self.train_X1[self.cnt])
        mms = MinMaxScaler()
        observations1 = mms.fit_transform(structure1)
        step = 0
        res1 = []
        test1 = []
        for tx1, ty1 in zip(self.test_X1[self.cnt], self.test_y1[self.cnt]):
            test1.append(ty1)
            # observations1, res1, actions1, step, node_ids1 = rl_step(self.tree_model, tx1, ty1, observations1, step,
            #                                                          res1, self.agent, self.pos_fb, self.neg_fb)
            observations1, res1, actions1, step, node_ids1 = self.rl_step_test(tx1, ty1, observations1, step, res1)

        states = compute_metrics(test1, res1)
        self.res_state.append(self.res_state_names, states)

        return states

    def rl_step_test(self, tx, ty, observations, step, res):
        mms = MinMaxScaler()
        actions = []
        node_ids = []

        uncertainty = self.tree_model.predict(tx, return_consistency=True, cut=False)
        # print(uncertainty)
        if np.mean(np.abs(uncertainty[1])) - 0.5 < 0.1:  # propose feedback (can be changed)
        # if 1:
        # if np.random.rand() <0.7:
            print('Client Test')
            for observation in observations:
                actions.append(self.agent.choose_action(observation))
            # print("action")
            # print(np.shape(actions))
            _, node_id = self.tree_model.update_structure(tx, actions, int(ty))
            node_ids.append(node_id)

            structure_ = []
            for atree in self.tree_model.trees:
                structure_.append(self.tree_model.record_tree(atree))
            observations_ = mms.fit_transform(structure_)  # normalize the tree structure
            # print(observations_)
            t = 0
            alpha = self.params['alpha']  # reward = alpha * global_reward + (1 - alpha) * regional_reward
            for observation, action, observation_ in zip(observations, actions, observations_):
                predy, pre_list = self.tree_model.predict(tx, cut=True)

                if predy == int(ty):
                    global_reward = 1
                    # print("positive feedback:")
                    # print(RL.pos_fb)
                else:
                    global_reward = -1
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
                    # not learn model, test what else performance difference
                    self.agent.learn()
                    pass
            observations = observations_
        predy, pre_list = self.tree_model.predict(tx, cut=True)

        res.append(predy)
        return observations, res, actions, step, node_ids



