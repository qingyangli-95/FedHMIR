import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # min-max normalized
from sklearn import metrics
import os


def load_data(file_list,
              train_ratio=0.3,
              contamination=0.2,
              seed=None):
    fnames = file_list
    train_data, test_data = [], []
    train_label, test_label = [], []

    rng = np.random.default_rng(seed)

    for af in fnames:
        raw_data = pd.read_excel(af, header=None).values
        train_len = int(train_ratio * len(raw_data))
        idx = np.arange(len(raw_data))
        # np.random.shuffle(idx)
        rng.shuffle(idx)
        dt = raw_data[idx[:train_len]]
        mms = MinMaxScaler()
        dt = mms.fit_transform(dt)
        train_data.append(dt)

        dl = raw_data[idx[train_len:]]
        dl = mms.transform(dl)
        test_data.append(dl)
        train_label.append([0] * train_len)
        test_label.append([0] * (len(raw_data) - train_len))

    for k in range(len(fnames)):
        data = train_data[k]
        dlen = len(data)
        idx = list(range(len(fnames)))
        idx.pop(k)
        # idx = np.random.choice(idx, size=int(dlen * contamination), replace=True)
        idx = rng.choice(idx, size=int(dlen * contamination), replace=True)
        other_data = []
        for ai in idx:
            other = train_data[ai]
            # p = np.random.choice(np.arange(len(other)))
            p = rng.choice(np.arange(len(other)))
            other_data.append(other[p])
        other_data = np.array(other_data)
        train_data[k] = np.vstack([train_data[k], other_data])
        train_label[k].extend([1] * int(dlen * contamination))

    for k in range(len(fnames)):
        data = test_data[k]
        dlen = len(data)
        idx = list(range(len(fnames)))
        idx.pop(k)
        # idx = np.random.choice(idx, size=dlen, replace=True)
        idx = rng.choice(idx, size=dlen, replace=True)
        other_data = []
        for ai in idx:
            other = test_data[ai]
            # p = np.random.choice(np.arange(len(other)))
            p = rng.choice(np.arange(len(other)))
            other_data.append(other[p])
        other_data = np.array(other_data)
        test_data[k] = np.vstack([test_data[k], other_data])
        test_label[k].extend([1] * dlen)


        index = [i for i in range(len(train_data[k]))]  # 初始树构建打乱
        # np.random.shuffle(index)
        rng.shuffle(index)
        train_data_shuffle = []
        train_label_shuffle = []
        for j in index:
            train_data_shuffle.append(train_data[k][j])
            train_label_shuffle.append(train_label[k][j])
        train_data[k] = train_data_shuffle
        train_label[k] = train_label_shuffle

        index = [i for i in range(len(test_data[k]))]  # update数据打乱
        # np.random.shuffle(index)
        rng.shuffle(index)
        test_data_shuffle = []
        test_label_shuffle = []
        for j in index:
            test_data_shuffle.append(test_data[k][j])
            test_label_shuffle.append(test_label[k][j])
        test_data[k] = test_data_shuffle
        test_label[k] = test_label_shuffle

    return train_data, train_label, test_data, test_label


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


def get_train_dataset(data_root, client_num, seed=None):
    np.random.seed(seed)
    dataset_size = len(os.listdir(data_root))
    print(data_root)
    print(dataset_size)
    if dataset_size <= 0:
        raise ValueError("dataset_size <= 0")
    indices = np.arange(1, dataset_size+1)
    np.random.shuffle(indices)
    if dataset_size % client_num == 0:
        indices = indices.reshape(client_num, -1)
    else:
        if dataset_size < client_num:
            raise ValueError("dataset_size < client_num")
        indices = indices[:dataset_size//client_num * client_num]
        indices = indices.reshape(client_num, -1)
    file_list = [list(map(lambda idx: os.path.join(data_root, str(idx)+".xlsx"), ind))\
        for ind in indices ]
    return file_list

def get_test_dataset(data_root):
    dataset_size = len(os.listdir(data_root))
    if dataset_size <= 0:
        raise ValueError("dataset_size <= 0")
    file_list = [os.path.join(data_root, str(idx) + ".xlsx") for idx in range(1, dataset_size+1)]
    return file_list


def compute_metrics(test, res):
    precision = metrics.precision_score(test, res)  # 看一下每一轮precision的变化情况
    auc = metrics.roc_auc_score(test, res)
    recall = metrics.recall_score(test, res)
    f1score = metrics.f1_score(test, res)
    tn, fp, fn, tp = metrics.confusion_matrix(test, res).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    output = np.array([precision, auc, recall, f1score, fpr, fnr], dtype=np.float)
    return output


def set_seed(seed):
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)

def rl_step(model, tx, ty, observations, step, res, RL):
    mms = MinMaxScaler()
    actions = []
    node_ids = []

    uncertainty = model.predict(tx, return_consistency=True, cut=False)
    # print(uncertainty)
    if np.mean(np.abs(uncertainty[1])) - 0.5 < 0.3:  # propose feedback (can be changed)
    # if 1:
    # if np.random.rand() <0.7:
        print('RL')
        for observation in observations:
            actions.append(RL.choose_action(observation))
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
        alpha = 0.9  # reward = alpha * global_reward + (1 - alpha) * regional_reward
        for observation, action, observation_ in zip(observations, actions, observations_):
            predy, pre_list = model.predict(tx, cut=True)

            if predy == int(ty):
                global_reward = 1
            else:
                global_reward = -1
            if pre_list[t] == int(ty):
                regional_reward = 1
            else:
                regional_reward = -1
            t += 1
            reward = alpha * global_reward + (1 - alpha) * regional_reward

            RL.store_transition(observation, action, reward, observation_)
            step += 1
            if step > 200 and (step % 10 == 0):
                # not learn model, test what else performance difference
                RL.learn()
                pass
        observations = observations_
    predy, pre_list = model.predict(tx, cut=True)

    res.append(predy)
    return observations, res, actions, step, node_ids

def test_step(model, tx, ty, observations, step, res, agent):
    mms = MinMaxScaler()
    actions = []
    node_ids = []
    print('RL test')

    uncertainty = model.predict(tx, return_consistency=True, cut=False)
    # if np.mean(np.abs(uncertainty[1])) - 0.5 < 0.3:
    # observations_ = None
    # if False:

    # seed0 = np.random.randint(0, 1e5, 1)
    # set_seed(1)
    for observation in observations:
        actions.append(agent.choose_action(observation))
    # actions = [1 for _ in range(10)]
    # set_seed(seed0)

    _, node_id = model.update_structure(tx, actions, int(ty))
    node_ids.append(node_id)

    structure_ = []
    for atree in model.trees:
        structure_.append(model.record_tree(atree))
    observations_ = mms.fit_transform(structure_)  # try to normalize the tree structure

    for observation, action, observation_ in zip(observations, actions, observations_):
        predy, pre_list = model.predict(tx, cut=True)

    predy, pre_list = model.predict(tx, cut=True)
    observations = observations_
    res.append(predy)
    return observations, res, actions, step, node_ids