'''
@qingyang li
run the whole model (train and test)
'''


import argparse

import os
import numpy as np

from functools import partial
import ray
ray.init(include_dashboard=False)
from ray import tune

from DDPG import DDPG
from data_preprocess import get_log_path, get_train_dataset, set_seed, make_dirs
from fl_server import Server
from fl_client import ClientVec


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_false')
args = parser.parse_args()

def create_client(params):
    ddpg = DDPG(params)
    return ddpg

'''
params = {
        "seed_idx": 0,
        "window_size": 10,
        "n_trees": 10,
        "depth": 5,
        'n_actions': None,  # number of actions
        'n_properties': 13,  # number of properties of one node, different from features of instance
        'n_nodes': None,

        # params for DDPG
        "LR_A": 0.001  # learning rate for actor
        "LR_C": 0.002  # learning rate for critic
        "GAMMA": 0.9  # reward discount
        "TAU": 0.01  # soft replacement
        “memory_capacity”： 10000  # size of replay buffer
        “batch_size": 32  # update batchsize
        
        # for exp
        "exp_num": exp_num,
        "log_path": log_path,  # log in total exp
    }
'''


class FLTrainable(tune.Trainable):
    def setup(self, config):
        import tensorflow as tf

        exp_num = 1
        data_path = 'D:/学习/博士工作/models/Multi-agent-RL-fl1 - 副本/Multi-agent-RL-fl/Data/Gait_authentication_preprocess'
        # data_path = "/home/yuanjiang/sda/work/pylab/Multi-agent-RL/Data/Data_50"
        out_path = os.path.join(data_path, "results")
        log_path = get_log_path(out_path, "federated")

        if not tune.is_session_enabled():
            raise ValueError("session not enabled")
        # log_path = "internal_log"
        # os.mkdir(log_path)
        config["seed"] = 1
        params = {
            "seed_idx": 0,
            "window_size": 10,
            "n_trees": config["n_trees"],
            "depth": config["depth"],
            'n_actions': None,  # number of actions
            'n_properties': 13,  # number of properties of one node, different from features of instance
            'n_nodes': None,

            # params for DDPG
            "layer_dims": config["dqn_layers"],
            'epsilon': config["epsilon"],
            'delta': config["delta"],
            'learning_rate': 1e-3,
            'reward_decay': 0.5,
            'e_greedy': 0.5,
            'e_greedy_increment': 0.2,
            'replace_target_iter': 100,
            'memory_size': 100,
            'batch_size': 30,

            # for exp
            "exp_num": exp_num,
            "log_path": log_path,  # log in total exp
            "seed": config["seed"],
            "data_path": data_path,
            "client_num": 2,
        }
        self.params = params

        params["n_nodes"] = 2 ** (params["depth"] + 1) - 1
        action_space = ['stay', 'increase', 'decrease', 'expand', 'collapse']
        params["n_actions"] = len(action_space)

        set_seed(config["seed"])

        # fp = open(os.path.join(log_path, "MARL_DNN.txt"), 'a')

        # params dict is used in two systems. one is assign a obj.params, the other
        # is the global params variable. Both point to the same single object. Thus when
        # server.train(params) or server.eval(params) is called, the params in main and
        # in client all changes.
        # params["fp"] = fp
        print("window_size: ", params["window_size"], "trees: ", params["n_trees"],
              "depth:", params["depth"])

        folders = ["train", "test"]
        make_dirs([os.path.join(log_path, folder) for folder in folders])

        # dirs = dataset(data_path)
        file_lists = get_train_dataset(data_path, client_num=params["client_num"],
                                       seed=params["seed"])
        agent_fn = partial(create_client, params=params)

        self.clients = ClientVec(client_num=2, datasets=file_lists, agent_fn=agent_fn, params=params)
        self.server = Server(self.clients, agent_fn, params)

        # server.load("./Data/results/exp-federated--000")
        # clients.load("./Data/results/exp-federated--000")

    def step(self):
        self.server.train(self.params)
        for _ in range(self.clients.get_train_steps()):
        # for _ in range(1):
            self.server.send_model()
            self.clients.train_step()
            self.server.aggregate_model()

        train_states = self.clients.log_summary(self.params)
        test_states = self.server.local_test(self.params["data_path"])

        # self.server.save(log_path)
        # self.clients.save(log_path)

        return {**train_states, **test_states}


def main(config):
    exp_num = 1
    data_path = './data/Gait_authentication_preprocess'
    out_path = os.path.join('./data', "results")
    log_path = get_log_path(out_path, "federated")
    # os.mkdir(log_path)
    config["seed"] = 1
    params = {
        "seed_idx": 0,
        "window_size": 10,
        "n_trees": config["n_trees"],
        "depth": config["depth"],
        'n_actions': None,  # number of actions
        'n_properties': 12,  # number of properties of one node, different from features of instance
        'n_nodes': None,

        # params for DDPG
        "LR_A": 0.001,  # learning rate for actor
        "LR_C": 0.002,  # learning rate for critic
        "GAMMA": 0.9,  # reward discount
        "TAU": 0.01,  # soft replacement
        "memory_capacity": 10000,  # size of replay buffer
        "batch_size": 32,  # update batchsize
        "a_bound": 0.5,

        # for exp
        "exp_num": exp_num,
        "log_path": log_path,  # log in total exp
        "seed": config["seed"],
        "data_path": data_path,
        "client_num": 2,
    }

    params["n_nodes"] = 2 ** (params["depth"] + 1) - 1
    action_space = ['stay', 'increase', 'decrease', 'expand', 'collapse']
    params["n_actions"] = len(action_space)

    set_seed(config["seed"])

    # fp = open(os.path.join(log_path, "MARL_DNN.txt"), 'a')

    # params dict is used in two systems. one is assign a obj.params, the other
    # is the global params variable. Both point to the same single object. Thus when
    # server.train(params) or server.eval(params) is called, the params in main and
    # in client all changes.
    # params["fp"] = fp
    print("window_size: ", params["window_size"], "trees: ", params["n_trees"],
          "depth:", params["depth"])

    folders = ["train", "test"]
    make_dirs([os.path.join(log_path, folder) for folder in folders])

    file_lists = get_train_dataset(data_path, client_num=params["client_num"],
                                   seed=config["seed"])
    agent_fn = partial(create_client, params=params)

    clients = ClientVec(client_num=params["client_num"], datasets=file_lists, agent_fn=agent_fn, params=params)
    server = Server(clients, agent_fn, params)

    # server.load("./Data/results/exp-federated--000")
    # clients.load("./Data/results/exp-federated--000")

    ##train
    server.train(params)
    for _ in range(clients.get_train_steps()):
        # for _ in range(1):
        server.send_model()
        clients.train_step()
        server.aggregate_model()
        server.save(params["out_path"])
        clients.save(params["out_path"])
    train_states = clients.log_summary(params)

    ##test
    test_states = server.local_test(params["data_path"])



    states = {**train_states, **test_states}
    for key, val in states.items():
        print("{}:\t{}".format(key, val))


if __name__ == "__main__":

    main(config={
        "depth": 5,
        "n_trees": 10,
    })

    # analysis = tune.run(
    #     FLTrainable,
    #     num_samples=1,
    #     local_dir='D:/学习/博士工作/models/Multi-agent-RL-fl1 - 副本/Multi-agent-RL-fl/Data/results',
    #     resources_per_trial={"cpu":1, "gpu":1},
    #     stop={"training_iteration": 2},
    #     config={
    #         "learning_rate": tune.grid_search([1e-2, 1e-3, 1e-4]),
    #         "depth": tune.grid_search([5,7,9,10,12]),
    #         "n_trees": tune.grid_search([5,10,15,20,25,30]),
    #         # "depth": tune.grid_search([5]),
    #         # "n_trees": tune.grid_search([10,]),
    #
    #
    #         "epsilon": tune.grid_search([0.5]), # 0.5, 1, 2, 4, 8, 12, 16
    #         # "delta": tune.grid_search([1e-5]), # 1e-3, 1e-4, 1e-5
    #         # "gradclip_bound": tune.grid_search([1]),
    #         "delta": tune.grid_search([1e-2, 1e-3, 1e-4, 1e-5]),
    #         "gradclip_bound": tune.grid_search([2,4,6,8,10]),
    #         # "seed": tune.randint(0, int(1e5))
    #     })
    #
    # print("Best config: ", analysis.get_best_config(metric="precisions", mode="max"))

    # # Get a dataframe for analyzing trial results.
    # df = analysis.results_df