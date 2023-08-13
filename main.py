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
from SPF import TreeModel
from HMIR import HMIR
from data_preprocess import get_log_path, get_train_dataset, set_seed, make_dirs
from fl_server import Server
from fl_client import ClientVec
import cProfile

# parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
# parser.add_argument('--train', dest='train', action='store_true', default=True)
# parser.add_argument('--test', dest='test', action='store_false')
# args = parser.parse_args()

def create_client(params):
    network = HMIR(params)
    return network


def main(config):
    data_path = './data/keystroke_CMU'
    out_path = os.path.join('./data', "results")
    log_path = get_log_path(out_path, "federated")
    config["seed"] = 1
    # os.mkdir(log_path)
    params = {
        "seed_idx": 0,
        "window_size": config["window_size"],
        "n_trees": config["n_trees"],
        "depth": config["depth"],
        'n_actions': 1,  # number of actions
        'n_properties': 12,  # number of properties of one node, different from features of instance
        'n_nodes': None,

        # params for actor-critic networks
        "LR_A": 0.001,  # learning rate for actor
        "LR_C": 0.001,  # learning rate for critic
        "GAMMA": 0.99,  # reward discount
        "TAU": 0.1,  # soft replacement
        "memory_capacity": 1000,  # size of replay buffer
        "batch_size": 512,  # update batchsize
        "a_bound": 1,
        "alpha": 0.9,
        "step": 100,

        # for exp
        "exp_num": 1,
        "log_path": log_path,  # log in total exp
        "data_path": data_path,
        "client_num": 10,
        "seed": config["seed"],
        "client_chosen": 3,
        "b": 0.3,
    }

    params["n_nodes"] = 2 ** (params["depth"] + 1) - 1
    # action_space = ['stay', 'increase', 'decrease', 'expand', 'collapse']
    # params["n_actions"] = len(action_space)

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
                                   seed=0)
    agent_fn = partial(create_client, params=params)

    clients = ClientVec(client_num=params["client_num"], datasets=file_lists, agent_fn=agent_fn, params=params, neg_fb=0, pos_fb=0,
                        tree=TreeModel(window_size=params["window_size"], n_trees=params["n_trees"], max_depth=params["depth"], min_depth=2, terminal_depth=4, adaptive=0.5, seed=10))
    server = Server(clients, agent_fn, params)

    # server.load("./Data/results/exp-federated--000")
    # clients.load("./Data/results/exp-federated--000")

    '''train'''
    server.train(params)
    # train_rounds = 10
    for _ in range(clients.get_train_steps()):
    # for _ in range(train_rounds):
    #     for _ in range(1):
        clients.train_step()
        # server.aggregate_model_fedavg()
        # server.aggregate_model_confi_random()
        server.aggregate_model_confi_selec()
        # server.save(params["out_path"])
        # clients.save(params["out_path"])
        server.send_model()

    train_states = clients.train_log_summary(params)
    # clients.plot_loss()

    '''client test'''
    # test_rounds = 1
    # for _ in range(test_rounds):
    for _ in range(clients.get_test_steps()):
        clients.test_step()
    client_test_states = clients.test_log_summary(params)

    '''server test'''

    test_states = server.local_test(params["data_path"])



    states = {**train_states, **client_test_states, **test_states}
    for key, val in states.items():
        print("{}:\t{}".format(key, val))


if __name__ == "__main__":

    cProfile.run('main(config={"window_size": 10,"depth": 7,"n_trees": 10,})')