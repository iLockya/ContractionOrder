import numpy as np
import torch
from math import log2
from functools import reduce
from utils import *

class TensorNetwork:
    def __init__(self, adjmatrix, eq, shapes):
        self.adjmatrix = adjmatrix
        self.eq = eq
        self.shapes = shapes

    @classmethod
    def from_eq(cls, eq, *tensor):
        shapes = [tn.shape for tn in tensor]
        in_, out_ = eq.split("->")
        in_ = in_.split(",")
        adjmatrix = {i:[0 for k in range(len(in_[i]))] for i in range(len(in_))}
        for i in range(len(in_)):
            for idx in in_[i]:
                if idx in out_:
                    adjmatrix[i][in_[i].index(idx)] = float("inf")
                else:
                    for j in range(i+1, len(in_)):
                        if idx in in_[j]:
                            adjmatrix[i][in_[i].index(idx)] = j
                            adjmatrix[j][in_[j].index(idx)] = i
        return TensorNetwork(adjmatrix, eq, shapes)

    def __repr__(self) -> str:
        TN_print = [
            "  Complete contraction:  {}\n".format(self.eq),
            "      Adjacency matrix     shapes\n"
        ]
        adjmatrix_print = [
            "{:<2}: {}  {}\n".format(i, self.adjmatrix[i], self.shapes[i]) for i in range(len(self.adjmatrix))
        ]
        return "".join(TN_print+adjmatrix_print)
        
                

class TreeNode:
    def __init__(self, id, subscripts, shape, parent = None, left = None, right = None, dtype = np.float):
        assert len(shape) == len(subscripts)
        self.id = id if type(id) == set else set([id])
        self.subscripts = subscripts
        self.shape = shape
        self.size_dict = dict(zip(subscripts, shape))
        self.parent = parent
        self.left = left
        self.right = right
        self.dtype = dtype
        self.node_sc = 1 if self.shape == () else reduce(lambda x,y: x*y, self.shape) 
        self.node_tc = 0
        self.sc, self.tc = self.node_sc, self.node_tc


    def __contains__(self, id):
        return id in self.id

    def __repr__(self) -> str:

        node_print = [
            "   id: " + repr(self.id) + "\n",
            " legs: " + repr(self.subscripts) + "\n",
            "   sc: {:.3e} elements\n".format(self.sc),
            "   tc: {:.3e}\n".format(self.tc)
        ]
        return "".join(node_print)

    @classmethod
    def merge_node(cls, node_l, node_r, trial = False):
        id = node_l.id  | node_r.id
        subscripts = str_symmetric_difference(node_l.subscripts, node_r.subscripts)
        size_dict = dict(node_l.size_dict, **node_r.size_dict)
        shape = tuple(size_dict[k] for k in subscripts)
        left = node_l
        right = node_r
        node_p = cls(id, subscripts, shape, left = node_l, right = node_r, dtype = node_l.dtype)
        node_p.node_tc = 0 if size_dict == {} else reduce(lambda x,y: x*y, size_dict.values())
        node_p.tc = node_p.node_tc + node_l.tc + node_r.tc
        node_p.sc = max(node_p.sc, node_p.left.sc, node_p.right.sc)
        if not trial:
            node_l.parent = node_p
            node_r.parent = node_p
        return node_p

    def sc_tc_update(self):
        #self.node_sc = 1 if self.shape == () else reduce(lambda x,y: x*y, self.shape) 
        #if self.isleaf():
        #    self.sc = self.node_sc
        #    self.node_tc = 0
        #else:
        self.sc = max(self.node_sc, self.left.sc, self.right.sc)
        #self.node_tc = 0 if self.size_dict == {} else reduce(lambda x,y: x*y, self.size_dict.values())
        self.tc = self.node_tc + self.left.tc + self.right.tc

    def isroot(self):
        return self.parent == None
    
    def isleaf(self):
        return (self.left == None and self.right == None)

class ContractionTree:
    def __init__(self, path, tree, seed = 0) -> None:
        self.path = path
        self.tree = tree
        self.sc_tc_update()
    def __repr__(self) -> str:

        tree_print = [
            "      sc: {:.3e} elements\n".format(self.sc),
            "      tc: {:.3e}\n".format(self.tc)
        ]

        return "".join(tree_print)


    @classmethod
    def from_info(cls, info, dtype = np.float):
        subscripts_list = info.input_subscripts.split(",")
        tensor_subscripts = {i:subscripts_list[i] for i in range(len(subscripts_list))}
        root_node = [TreeNode(i, subscripts_list[i], tuple(info.size_dict[k] for k in tensor_subscripts[i]),\
             dtype=dtype) for i in range(len(subscripts_list))]
        for (l, r) in info.path:
            node_l, node_r = root_node[l], root_node[r]
            node_p = TreeNode.merge_node(node_l, node_r)
            root_node.append(node_p)
            root_node.remove(node_l)
            root_node.remove(node_r)

        if len(root_node) == 1:
            tree = root_node[0]
        else:
            tree = root_node
        return cls(path = info.path, tree = tree)

    def sc_tc_update(self):
        self.sc = self.tree.sc
        self.tc = self.tree.tc

    def local_transform(self, node, beta):
        # case 0: A * (B * C) -> B * (A * C)
        # case 1: A * (B * C) -> C * (B * A)
        # case 2: (A * B) * C -> (C * B) * A
        # case 3: (A * B) * C -> (A * C) * B
        if not node.left.isleaf() and not node.right.isleaf():
            case = np.random.randint(4)
        elif not node.left.isleaf() and node.right.isleaf():
            case = np.random.randint(2) + 2
        elif node.left.isleaf() and not node.right.isleaf():
            case = np.random.randint(2)

        case = 0
        if case == 0: # A * (B * C) -> B * (A * C)
            A, B, C = node.left, node.right.left, node.right.right
            node_new = TreeNode.merge_node(B, TreeNode.merge_node(A, C, trial = True), trial = True)
            sc_new = max(self.sc, node_new.sc)
            tc_new = self.tc - node.tc + node_new.tc
            delta = target_function(sc_new, tc_new, mem = 0) - target_function(self.sc, self.tc, mem = 0)
            if np.random.rand() <= min(1, np.exp(-beta*delta)):  # Accepted
                B.parent, A.parent, C.parent, node_new.right.parent = node_new, node_new.right, node_new.right, node_new
                if not node.isroot():
                    node_new.parent = node.parent
                    if node.parent.left == node:
                        node.parent.left = node_new
                    else:
                        node.parent.right = node_new
                node_ = node_new
                while not node_.isroot():
                    node_ = node_.parent
                    node_.sc_tc_update()
        elif case == 1: # A * (B * C) -> C * (B * A)
            A, B, C = node.left, node.right.left, node.right.right
            node_new = TreeNode.merge_node(C, TreeNode.merge_node(B, A, trial = True), trial = True)
            sc_new = max(self.sc, node_new.sc)
            tc_new = self.tc - node.tc + node_new.tc
            delta = target_function(sc_new, tc_new, mem = 0) - target_function(self.sc, self.tc, mem = 0)
            if np.random.rand() <= min(1, np.exp(-beta*delta)):
                C.parent, B.parent, A.parent, node_new.right.parent = node_new, node_new.right, node_new.right, node_new
                if not node.isroot():
                    node_new.parent = node.parent
                    if node.parent.left == node:
                        node.parent.left = node_new
                    else:
                        node.parent.right = node_new
                node_ = node_new
                while not node_.isroot():
                    node_ = node_.parent
                    node_.sc_tc_update()
        elif case == 2: # (A * B) * C -> (C * B) * A
            A, B, C = node.left.left, node.left.right, node.right
            node_new = TreeNode.merge_node(TreeNode.merge_node(C, B, trial = True), trial = True)
            sc_new = max(self.sc, node_new.sc)
            tc_new = self.tc - node.tc + node_new.tc
            delta = target_function(sc_new, tc_new, mem = 0) - target_function(self.sc, self.tc, mem = 0)
            if np.random.rand() <= min(1, np.exp(-beta*delta)):
                A.parent, B.parent, C.parent, node_new.left.parent = node_new, node_new.left, node_new.left, node_new
                if not node.isroot():
                    node_new.parent = node.parent
                    if node.parent.left == node:
                        node.parent.left == node
                    else:
                        node.parent.right = node_new
                node_ = node_new
                while not node_.isroot():
                    node_ = node_.parent
                    node_.sc_tc_update()
        else: # (A * B) * C -> (A * C) * B
            A, B, C = node.left.left, node.left.right, node.right
            node_new = TreeNode.merge_node(TreeNode.merge_node(A, C, trial = True), trial = True)
            sc_new = max(self.sc, node_new.sc)
            tc_new = self.tc - node.tc + node_new.tc
            delta = target_function(sc_new, tc_new, mem = 0) - target_function(self.sc, self.tc, mem = 0)
            if np.random.rand() <= min(1, np.exp(-beta*delta)):
                A.parent, B.parent, C.parent, node_new.left.parent = node_new.left, node_new, node_new.left, node_new
                if not node.isroot():
                    node_new.parent = node.parent
                    if node.parent.left == node:
                        node.parent.left == node
                    else:
                        node.parent.right = node_new
                node_ = node_new
                while not node_.isroot():
                    node_ = node_.parent
                    node_.sc_tc_update()


        




def target_function(sc, tc, mem):
    return log2(tc)

if __name__ == "__main__":
    import opt_einsum as oe
    eq, shapes = oe.helpers.rand_equation(n=50, reg=5, seed=42, d_max=2)
    arrays = [np.random.uniform(size=s) for s in shapes]
    path, info = oe.contract_path(eq, *arrays, optimize='greedy')
    #info.path = [(0, 1) for i in range(len(info.path))]
    ctree = ContractionTree.from_info(info)
    print(ctree.tree)
    #ctree.local_transform(ctree.tree.right, 10)
    #print(ctree.tree)
            
