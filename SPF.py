'''
@qingyang li
Space-prationing forest with update mechanism
Environment contains tree model and human
tree model provides state and identification result
human provides reward (feedback).
'''


import numpy as np
from Utilities import IncrementalVar, entropy_freqs
from itertools import chain

EPSILON = np.finfo(np.float32).eps

class Node:
    def __init__(self, node_id, level, parent=None,
                 inst_ref=0, inst_lat=0, is_leaf=False,
                 norm_ref=0, norm_lat=0, abno_ref=0,
                 abno_lat=0):
        self.id = node_id
        self.level = level
        self.is_leaf = is_leaf
        self.feature = 1000  # 由于要将整棵树存储并作为输入，因此初始值不能为空实际特征数更大的一个数字
        self.split = 1000  # 同上，初始值不能为空，设置为一个较大的数字
        self.is_terminal = False
        self.inst_ref = inst_ref
        self.inst_lat = inst_lat
        self.norm_ref = norm_ref
        self.norm_lat = norm_lat
        self.abno_ref = abno_ref
        self.abno_lat = abno_lat
        self.parent = parent
        self.left = None
        self.right = None
        self.traverse = False  # a flag to record if the instance has traversed the node

    @property
    def normal(self):
        return self.norm_ref + self.norm_lat

    @property
    def abnormal(self):
        return self.abno_ref + self.abno_lat

    def update(self, adaptive=0):
        if adaptive < 0 or adaptive > 1:
            raise ValueError('adaptive must in [0, 1]')
        self.inst_ref = adaptive * self.inst_ref + (1 - adaptive) * self.inst_lat
        self.inst_lat = 0
        self.norm_ref = adaptive * self.norm_ref + (1 - adaptive) * self.norm_lat
        self.norm_lat = 0
        self.abno_ref = adaptive * self.abno_ref + (1 - adaptive) * self.abno_lat
        self.abno_lat = 0

    def __call__(self, x):
        self.n += 1
        new_mean = self.im(x)
        self.sn = self.sn + (x - self.mean) * (x - new_mean)
        self.var = self.sn / self.n
        self.mean = new_mean
        return self.var

    def __repr__(self):
        if self.is_terminal:
            return "%d *\n%d, %.3f\n%.1f, %d\n%.1f, %d\n%.1f, %d" % (self.id,
                                                                     self.feature if self.feature else 0,
                                                                     self.split if self.split else 0,
                                                                     self.inst_ref, self.inst_lat, self.norm_ref,
                                                                     self.norm_lat, self.abno_ref, self.abno_lat)
        else:
            return "%d\n%d, %.3f\n%.1f, %d\n%.1f, %d\n%.1f, %d" % (self.id,
                                                                   self.feature if self.feature else 0,
                                                                   self.split if self.split else 0,
                                                                   self.inst_ref, self.inst_lat, self.norm_ref,
                                                                   self.norm_lat, self.abno_ref, self.abno_lat)
    def traverse_repr(self):
        res = self.__repr__()
        res += "\n" + self.left.traverse_repr()
        res += "\n" + self.right.traverse_repr()
        res += "\n"
        return res

def compare_nodes(node1, node2):
    # print(vars(node1), vars(node1) == vars(node2))
    attrs = filter(lambda a: not (a.startswith("__") or callable(getattr(node1, a))),
                   dir(node1))
    for attr in list(attrs):
        if attr in ["left", "right"]:
            continue
        # print(attr, getattr(node1, attr))
        if not (getattr(node1, attr) == getattr(node2, attr)):
            print(getattr(node1, attr), getattr(node2, attr))
            return False
    return True

def Node_test():
    n = 2
    nodes = [Node(n,n) for i in range(n)]
    # print(type(nodes[0].abnormal))
    for node in nodes:
        print(node)
    print(compare_nodes(*nodes))

# if __name__ == "__main__":
#     Node_test()

N_Trees = 15  # number of trees in a model
MaxDepth = 20  # maximum depth of a tree
MinDepth = 3  # minimum depth of a tree to avoid collapsing the root node when take 'collapse' action
TerDepth = 7  # terminal depth of a tree for the beginning of the growth, to ensure the space for 'growth' action
WinSize = 250  # a time window is set for updating inst_ref and inst_lat
Adaptive = 0.95  # for updating inst_ref and inst_lat, inst_ref = adaptive * inst_ref + (1 - adaptive) * inst_lat
Contam = 0.1  # contamination is for calculating threshold


class TreeModel:
    def __init__(self, n_trees=N_Trees, max_depth=MaxDepth, min_depth=MinDepth,
                 terminal_depth=TerDepth, window_size=WinSize, adaptive=Adaptive,
                 contamination=Contam, seed=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.terminal_depth = min(terminal_depth, max_depth - 1)
        self.min_depth = min(min_depth, max_depth)
        self.window_size = window_size
        self.contamination = contamination
        self.adaptive = adaptive
        self.mass_mv = [IncrementalVar() for _ in range(n_trees)]
        # mean and var of the mass in every trees
        self.terminal_entropys = []
        self.trees = []
        # entropy of inst_ref on every leaf nodes
        self.__n_nodes = 0
        self.__ins_count = 0
        self.threshold = 0

        self.rng = np.random.default_rng(seed)

    def set_contamination(self, contamination):
        self.contamination = contamination
        self.threshold = np.sort(self.s_scores)[int(self.window_size * self.contamination)]

    @property
    def node_id(self): # need to be recorded in the state
        self.__n_nodes += 1
        return self.__n_nodes

    def init_work_space(self, ndims):
        # sqs = np.random.uniform(size=ndims)
        sqs = self.rng.uniform(size=ndims)
        work_range = 2 * np.maximum(sqs, 1 - sqs)
        # work_range = 1.5 * np.maximum(sqs, 1-sqs)
        maxqs = sqs + work_range
        minqs = sqs - work_range
        # minqs = 0
        # maxqs = ndims
        return sqs, minqs, maxqs

    def choose_dim_by_len(self, mins, maxs):
        lens = np.abs(maxs - mins)
        plens = lens / np.sum(lens)
        feature = self.rng.choice(len(mins), p=plens)
        # feature = np.random.choice(len(mins), p=plens)
        # feature = np.random.choice(maxs, p=plens)
        return feature

    def tree_growth(self, node, mins, maxs): # 树的构造
        if node.level == self.terminal_depth:
            node.is_terminal = True
        if node.level >= self.max_depth:
            node.is_leaf = True
            return
        # feature = np.random.choice(len(mins))
        # split = (mins[feature] + maxs[feature])/2
        feature = self.choose_dim_by_len(mins, maxs)
        split = self.rng.uniform(low=mins[feature], high=maxs[feature])
        # split = np.random.uniform(low=mins[feature], high=maxs[feature])
        # feature = np.random.choice(len(mins))
        # split = (mins[feature] + maxs[feature])/2
        node.feature = feature
        node.split = split
        newmins = mins.copy()
        newmins[feature] = split
        newmaxs = maxs.copy()
        newmaxs[feature] = split
        node.left = Node(self.node_id, node.level + 1)
        node.left.parent = node
        node.right = Node(self.node_id, node.level + 1)
        node.right.parent = node
        self.tree_growth(node.left, mins, newmaxs)
        self.tree_growth(node.right, newmins, maxs)

    def record_tree(self, node): # traverse，record nodes
        if node is None:
             return ''
        queue = [node]
        str = []
        while queue:
            current = queue.pop(0)
            str.append(current)
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
        # print(len(str))

        structure = []
        for item in str:
            info = [item.level, item.is_leaf, item.feature, item.split, item.is_terminal,
                    item.inst_ref, item.inst_lat, item.norm_ref, item.norm_lat, item.abno_ref, item.abno_lat,
                    item.traverse]
            # info = [item.inst_ref, item.inst_lat, item.norm_ref, item.norm_lat, item.abno_ref, item.abno_lat,
                    # item.traverse]
            structure.append(info)  # node_num * info_dim
        # print(np.shape(structure))
        s = list(chain(*structure))  # 将node_info矩阵展开为向量
        # print(s)
        return np.array(s)

    def traverse_record_terminals(self, node, leaves):
        if node.is_terminal:
            leaves.append(node)
            return
        self.traverse_record_terminals(node.left, leaves)
        self.traverse_record_terminals(node.right, leaves)


    def calc_terminal_entropy(self):
        self.terminal_entropys = []
        for root in self.trees:
            terminal_list = []
            self.traverse_record_terminals(root, terminal_list)
            entropy = entropy_freqs([al.inst_ref * (2 ** al.level) for al in terminal_list])
            # self.terminal_entropys.append(1 - entropy / self.max_depth)
            self.terminal_entropys.append(1 - entropy / self.max_depth)
        z = sum(self.terminal_entropys)
        self.terminal_entropys = [aen / z for aen in self.terminal_entropys]
        return

    def traverse(self, node, x,
                 update_type='inst_lat',
                 update=True,
                 return_node=None):
        """
        @update_type: 'inst_ref', 'inst_lat', 'norm_feed', 'abno_feed'
        """
        if update and update_type == 'inst_ref':
            node.inst_ref += 1
        if update and update_type == 'inst_lat':
            node.inst_lat += 1

        if update and update_type == 'norm_feed':
            node.norm_lat += 1
        if update and update_type == 'abno_feed':
            node.abno_lat += 1
            node.inst_lat = max(0, node.inst_lat - 1)

        if node.is_terminal:
            return_node = node
            node.traverse = True
            # return node
        if node.is_leaf:
            if return_node:
                node = return_node
            node.traverse = True
            return node
        feature = node.feature
        if x[feature] < node.split:
            node.traverse = True
            return self.traverse(node.left, x, update_type, update, return_node)
        else:
            node.traverse = True
            return self.traverse(node.right, x, update_type, update, return_node)

    def mass_on_nodes(self, nodes):
        masses = []
        for k, anode in enumerate(nodes):
            mass = anode.inst_ref * (2 ** anode.level)
            mass_mv = self.mass_mv[k]
            mass_mv(mass)
            masses.append(mass)
        return masses

    def score_on_masses(self, masses):  #
        scores = []
        consistencies = []
        for masslist in masses:
            raw_scores = [1 - self.sigmoid(amass, amv)
                          for amass, amv in zip(masslist, self.mass_mv)]
            avg_score = np.mean(raw_scores)
            scores.append(avg_score)
        return scores

    def score_on_nodes(self, nodes, return_score_list=True,
                       update_mass=True):
        scores = []
        errors = []
        for k, anode in enumerate(nodes):
            mass = anode.inst_ref * (2 ** anode.level)
            mass_mv = self.mass_mv[k]
            if update_mass:
                mass_mv(mass)
            smi = self.sigmoid(mass, mass_mv)
            scores.append(1 - smi)
            aini = anode.normal + anode.abnormal
            if aini > 0:
                expected = anode.normal / aini
            else:
                expected = smi
            error = np.abs(expected - smi)
            errors.append(error)
        # consistency = entropy_freqs(scores)/np.log2(len(self.trees))
        # consistency = np.std(scores)
        consistency = np.mean(errors)
        scores_ = np.array(scores)
        avg_score = np.mean(scores_)
        # terminal inst_ref 的diversity，是否有冲突?
        # raw_score = np.average(masses, weights=self.terminal_entropys)
        # 论文
        if return_score_list:
            return avg_score, consistency, scores_
        return avg_score, consistency

    def sigmoid(self, x, mass_mv):  # 论文
        if mass_mv.var < 1e-10:
            mass_mv(0)
        # gamma = np.sqrt(3 * mass_mv.var) / np.pi
        gamma = np.pi * np.sqrt(mass_mv.var/ 3)
        return 1.0 / (1.0 + np.exp((-x + mass_mv.mean) / gamma))

    # 数据fit到树结构当中
    def fit(self, X):
        X = np.array(X)
        N, M = X.shape
        self.trees = []
        # aroot = Node(node_id=0, level=1)
        for i in range(self.n_trees):
            _, mins, maxs = self.init_work_space(M)
            aroot = Node(self.node_id, 0)
            self.tree_growth(aroot, mins, maxs)
            for x in X:
                self.traverse(aroot, x, update_type='inst_ref')
            self.trees.append(aroot) # 存储各个树的根节点
        self.calc_terminal_entropy()
        masses = np.array([self.mass_on_nodes([self.traverse(atree, x, update=False)
                                               for atree in self.trees]) for x in X])
        self.s_scores = self.score_on_masses(masses)
        self.threshold = np.sort(self.s_scores)[int(N * (1 - self.contamination))]
        # 记录每棵树的structure
        structure = []
        for atree in self.trees:
            structure.append(self.record_tree(atree))
        return self, structure

    def update_tree(self, node): # update node.ref and node.lat when time window is full
        node.update(adaptive=self.adaptive)
        if node.is_leaf:
            return
        self.update_tree(node.left)
        self.update_tree(node.right)

    # 根据fit的树来预测
    # 改进：同时得到每棵树的结果的information_entropy，与最终预测结果和feedback的比较，一起作为global_reward
    def predict(self, x, cut=True,
                scale_score = False,
                return_consistency=False):
        self.__ins_count += 1
        if self.__ins_count >= self.window_size:
            for atree in self.trees:
                self.update_tree(atree)
            self.calc_terminal_entropy()
            self.__ins_count = 0
        terminals = [self.traverse(atree, x, update_type='inst_lat') for atree in self.trees]
        score, consistency, score_list = self.score_on_nodes(terminals)
        # calculate information entropy of trees
        # num0 = 0
        # num1 = 0
        pre_list = []  # 每棵树的预测结果，和groundtruth作比较，作为regional_reward
        for s in score_list:
            pre_list.append(int(s > self.threshold))
        #     if s > self.threshold:
        #         num1 += 1
        #     else:
        #         num0 += 1
        # p_1 = num0 / (num0 + num1)
        # p_2 = num1 / (num0 + num1)
        # information_entropy = - (p_1 * np.log(p_1) + p_2 * np.log(p_2))

        if return_consistency:
            score = [score, consistency]
        if not cut:
            return score
        else:
            return int(score > self.threshold), pre_list


    # def expand_node(self, node, g, r):
    #     if node.is_leaf:
    #         adjust_rate = 0.5
    #         node.inst_ref -= adjust_rate * (g + r) * (2 ** node.level)
    #         return self
    #     node.is_terminal = False
    #     node.left.is_terminal = True
    #     node.right.is_terminal = True
    #     return self
    #
    # def collapse_node(self, node, g, r):
    #     if node.level <= self.min_depth:
    #         adjust_rate = 0.5
    #         node.inst_ref += adjust_rate * (g + r) * (2 ** node.level)
    #         return self
    #     parent = node.parent
    #     if node.id == parent.left.id:
    #         brother = parent.right
    #     else:
    #         brother = parent.left
    #     terminals = []
    #     self.traverse_record_terminals(brother, terminals)
    #     for aterm in terminals:
    #         aterm.is_terminal = False
    #     node.is_terminal = False
    #     parent.is_terminal = True
    #     return self

    def expand_node(self, node, t, y, mass_mv):
        if node.is_leaf:
            return False
        gl, rl = self.node_derivative(node.left, t, y, mass_mv)
        gr, rr = self.node_derivative(node.right, t, y, mass_mv)
        if rl < -EPSILON and rr < -EPSILON:
            node.is_terminal = False
            node.left.is_terminal = True
            node.right.is_terminal = True
            return True
        return False

    def collapse_node(self, node, t, y, mass_mv):
        if node.level <= self.min_depth:
            return False
        parent = node.parent
        if node.id == parent.left.id:
            brother = parent.right
        else:
            brother = parent.left
        terminals = []
        self.traverse_record_terminals(brother, terminals)
        derivatives = np.array([self.node_derivative(aterm, t, y, mass_mv)
                                for aterm in terminals])
        rs = derivatives[:, 1]
        if all(rs > EPSILON):
            for aterm in terminals:
                aterm.is_terminal = False
            node.is_terminal = False
            parent.is_terminal = True
            return True
        return False

    def node_derivative(self, node, t, y, mass_mv):
        mass = node.inst_ref * (2 ** node.level)
        smi = self.sigmoid(mass, mass_mv)
        global_de = smi * (1 - smi) * (y - t) / (y * (1 - y) * self.n_trees)
        ni, ai = node.normal, node.abnormal
        local_de = ni - (ai + ni) * smi
        return global_de, local_de

    def update_structure(self, x, actions, label):
        node_id = []
        if label == 0:
            feed_type = 'norm_feed'
        else:
            feed_type = 'abno_feed'
        terminals = [self.traverse(atree, x, update_type=feed_type) for atree in self.trees]
        y, consistency, s = self.score_on_nodes(terminals, return_score_list=True, update_mass=False)
        t = int(label)
        global_derivative = s * (1 - s) * (y - t) / (y * (1 - y) * len(terminals))
        nas = np.array([[aterminal.normal, aterminal.abnormal] for aterminal in terminals])
        ni = nas[:, 0]
        ai = nas[:, 1]
        regional_derivative = ni - (ai + ni) * s
        # adjust_rate = 0.1

        actions = np.squeeze(actions)
        # print(np.shape(terminals))
        # print(np.shape(actions))
        for action, node, g, r, amv in zip(actions, terminals, global_derivative, regional_derivative, self.mass_mv):
            node_id.append(node.id)
            node.inst_ref += action * (g + r) * (2 ** node.level)
            # if action == 0: # stay
            #     continue
            # # if action == 'increase':
            # if action == 1: # increase
            #     # continue
            #     # node.inst_ref += adjust_rate * (g + r) * (2 ** node.level)
            #     node.inst_ref += (g + r) * (2 ** node.level)
            #
            # if action == 2: # decrease
            #     # continue
            #     # node.inst_ref -= adjust_rate * (g + r) * (2 ** node.level)
            #     node.inst_ref -= (g + r) * (2 ** node.level)
            #
            # if action == 3: # expand
            #     try_expand = self.expand_node(node, t, y, amv)
            #     # self.expand_node(node, g, r)
            #     if not try_expand:
            #         # node.inst_ref -= adjust_rate * (2 ** node.level)
            #         # node.inst_ref -= adjust_rate * (g + r) * (2 ** node.level)
            #         continue
            # if action == 4: # collapse
            #     try_collapse = self.collapse_node(node, t, y, amv)
            #     # self.collapse_node(node, g, r)
            #     if not try_collapse:
            #         # node.inst_ref += adjust_rate * (2 ** node.level)
            #         # node.inst_ref += adjust_rate* (g + r) * (2 ** node.level)
            #         continue
        return self, node_id

    def update_structure_no_actions(self, x, label):
        node_id = []
        if label == 0:
            feed_type = 'norm_feed'
        else:
            feed_type = 'abno_feed'
        terminals = [self.traverse(atree, x, update_type=feed_type) for atree in self.trees]
        y, consistency, s = self.score_on_nodes(terminals, return_score_list=True, update_mass=False)
        return self, node_id

    def __str__(self):
        attrs = filter(lambda a: not a.startswith("__") and not callable(getattr(self, a)), dir(self))
        attr_list = list(attrs)
        res = ""
        for attr in attr_list:
            res += attr + ":\t" + str(getattr(self, attr)) + "\n"
        for t in self.trees:
            res += str(t) + "\n"

        return res

def compare_tree(tree1, tree2):
    # print(tree1)
    # print(tree2)
    # print(compare_nodes(tree1, tree2))
    # print("\n")
    if tree1 is None:
        if tree2 is None:
            return True
        else:
            return False
    elif tree2 is None:
        return False
    if not compare_nodes(tree1, tree2):
        return False
    if not compare_tree(tree1.left, tree2.left):
        return False
    if not compare_tree(tree1.right, tree2.right):
        return False
    return True


def compare_trees(trees1, trees2):
    for t1, t2 in zip(trees1, trees2):
        if not compare_tree(t1, t2):
            return False
    return True

def compare_model(obj1, obj2):
    attrs1 = list(filter(lambda a: not a.startswith("__") and not callable(getattr(obj1, a)), dir(obj1)))
    attrs2 = list(filter(lambda a: not a.startswith("__") and not callable(getattr(obj2, a)), dir(obj2)))
    assert attrs1 == attrs2
    for attr in attrs1:
        # print(attr)
        if attr == "trees":
            if not compare_trees(obj1.trees, obj2.trees):
                return False
        elif attr == "rng":
            continue
        elif getattr(obj1, attr) != getattr(obj2, attr):
            print(attr)
            print(getattr(obj1, attr), getattr(obj2, attr))
            return False

    return True

def Tree_test():
    models = [TreeModel(n_trees=2, max_depth=2, min_depth=2, terminal_depth=2, window_size=10, seed=10)
             for _ in range(2)]

    X = np.random.rand(1, 494)
    for m in models:
        m.fit(X)
        np.random.randn(10)

    # print(getattr(models[0], "_TreeModel__n_nodes"))
    # print(getattr(models[1], "_TreeModel__n_nodes"))

    print(models[0])
    print(models[1])
    print(compare_model(*models))


# def char2int(c):
#     return ord(c) - ord("a") + 1
#
# # compute the dimension of features in dataset, from xls RZ to number
# def compute_dim():
#     r = char2int("r")
#     z = char2int("z")
#     print(r*26 + z)


if __name__ == "__main__":
    Tree_test()
    # compute_dim()