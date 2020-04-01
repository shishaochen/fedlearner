# Copyright 2020 The FedLearner Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8

import math
import numpy as np
from google.protobuf import text_format

from fedlearner.model.tree.loss import LogisticLoss
from fedlearner.model.crypto import paillier, fixed_point_number
from fedlearner.common import tree_model_pb2 as tree_pb2

BST_TYPE = np.float32
PRECISION = 1e38
EXPONENT = math.floor(math.log(PRECISION, fixed_point_number.FixedPointNumber.BASE))
KEY_NBITS = 1024
CIPHER_NBYTES = (KEY_NBITS * 2)//8

def _compute_histogram(values, ptrs, leaf_nodes, assignment, num_bins, bin_size, zero=0.0):
    hists = {}
    num_nodes = len(leaf_nodes)
    for fid, ptr in ptrs.items():
        num_examples = len(ptr)
        hist = [[zero for _ in range(num_bins)] for _ in range(num_nodes)]
        for i in range(num_examples):
            hist[assignment[ptr[i]]][i // bin_size] += values[ptr[i]]
        hists[fid] = hist

    return hists


def _find_split(grad_hists, hess_hists, lam):
    max_gain = -1
    max_fid = None
    split_point = None
    left_weight = None
    right_weight = None
    for fid, grad_hist in grad_hists.items():
        hess_hist = hess_hists[fid]
        sum_g = sum(grad_hist)
        sum_h = sum(hess_hist)
        left_g = 0.0
        left_h = 0.0
        for i in range(len(grad_hist) - 1):
            left_g += grad_hist[i]
            left_h += hess_hist[i]
            right_g = sum_g - left_g
            right_h = sum_h - left_h
            gain = left_g*left_g/(left_h + lam) + \
                   right_g*right_g/(right_h + lam) - \
                   sum_g*sum_g/(sum_h + lam)
            if gain > max_gain:
                max_gain = gain
                max_fid = fid
                split_point = i + 1
                left_weight = - left_g/(left_h + lam)
                right_weight = - right_g/(right_h + lam)
    return max_gain, max_fid, split_point, left_weight, right_weight


def _split_node(tree, node, bin_size, max_fid=None, sorted_features=None, split_point=None, left_weight=None, right_weight=None):
    if max_fid is not None:
        node.is_owner = True
        node.feature_id = max_fid
        feature = sorted_features[max_fid]
        node.threshold = (feature[split_point*bin_size] + feature[split_point*bin_size + 1])/2.0
    else:
        node.is_owner = False

    node.left_child = len(tree.nodes)
    left_child = tree_pb2.RegressionTreeNodeProto(node_id=node.left_child)
    left_child.parent = node.node_id
    if left_weight is not None:
        left_child.weight = left_weight
    tree.nodes.append(left_child)

    node.right_child = len(tree.nodes)
    right_child = tree_pb2.RegressionTreeNodeProto(node_id=node.right_child)
    right_child.parent = node.node_id
    if right_weight is not None:
        right_child.weight = right_weight
    tree.nodes.append(right_child)

    return left_child, right_child


def _send_public_key(bridge, public_key):
    msg = tree_pb2.EncryptedNumbers()
    msg.ciphertext.append(public_key.n.to_bytes(KEY_NBITS//8, 'little'))
    bridge.send_proto(bridge.current_iter_id, 'public_key', msg)

def _receive_public_key(bridge):
    msg = tree_pb2.EncryptedNumbers()
    bridge.receive_proto(bridge.current_iter_id, 'public_key').Unpack(msg)
    return paillier.PaillierPublicKey(int.from_bytes(msg.ciphertext[0], 'little'))

def _encode_encrypted_numbers(numbers):
    return [i.ciphertext().to_bytes(CIPHER_NBYTES, 'little') for i in numbers]

def _encrypt_numbers(public_key, numbers):
    return _encode_encrypted_numbers([public_key.encrypt(i, PRECISION) for i in numbers])

def _decrypt_number(private_key, numbers):
    return [private_key.decrypt(i) for i in numbers]

def _from_ciphertext(public_key, ciphertext):
    return [
        paillier.PaillierEncryptedNumber(
            public_key, int.from_bytes(i, 'little'), EXPONENT)
        for i in ciphertext]

def _encrypt_and_send_numbers(bridge, name, public_key, numbers):
    msg = tree_pb2.EncryptedNumbers()
    msg.ciphertext.extend(_encrypt_numbers(public_key, numbers))
    bridge.send_proto(bridge.current_iter_id, name, msg)

def _receive_encrypted_numbers(bridge, name, public_key):
    msg = tree_pb2.EncryptedNumbers()
    bridge.receive_proto(bridge.current_iter_id, name).Unpack(msg)
    return _from_ciphertext(public_key, msg.ciphertext)

def _send_histograms(bridge, name, hists):
    msg = tree_pb2.Histograms()
    for fid, hist in hists.items():
        msg.feature_ids.append(fid)
        ciphertext = _encode_encrypted_numbers(sum(hist, []))
        msg.hists.append(
            tree_pb2.EncryptedNumbers(ciphertext=ciphertext))
    bridge.send_proto(bridge.current_iter_id, name, msg)

def _receive_and_decrypt_histogram(bridge, name, public_key, private_key, num_bins, num_nodes):
    msg = tree_pb2.Histograms()
    bridge.receive_proto(bridge.current_iter_id, name).Unpack(msg)
    hists = {}
    for i, feature_id in enumerate(msg.feature_ids):
        numbers = _from_ciphertext(public_key, msg.hists[i].ciphertext)
        numbers = _decrypt_number(private_key, numbers)
        hists[feature_id] = [numbers[num_bins*j:num_bins*(j+1)] for j in range(num_nodes)]
    return hists

class BoostingTreeEnsamble(object):

    def __init__(self, proto):
        self._proto = proto
        self._params = self._proto.params
        self._loss = LogisticLoss()

    @classmethod
    def from_config(cls, num_round, max_depth=6, lam=1.0, eta=0.3, sketch_eps=0.03):
        proto = tree_pb2.BoostingTreeEnsambleProto(
            params=tree_pb2.BoostingParamsProto(
                num_round=num_round,
                max_depth=max_depth,
                lam=lam,
                eta=eta,
                sketch_eps=sketch_eps))
        return cls(proto)

    @classmethod
    def from_saved_model(cls, path):
        with open(path, 'r') as fin:
            model = tree_pb2.BoostingTreeEnsamble()
            text_format.Parse(fin.read(), model)
        return cls(model)

    def save_model(self, path):
        with open(path, 'w') as fout:
            fout.write(text_format.MessageToString(self._params))

    def batch_predict(self, bridge, example_ids, features):
        if bridge.role == 'leader':
            return self._batch_predict_leader(bridge, example_ids, features)
        else:
            return self._batch_predict_follower(bridge, example_ids, features)

    def _batch_predict_leader(self, bridge, example_ids, features):
        N = len(example_ids)
        fx = np.zeros(N, dtype=np.float32)
        for tree in self._proto.trees:
            assignment = np.zeros(N, dtype=np.int32)
            leaf_count = 0
            while leaf_count != N:
                bridge.start(bridge.new_iter_id())
                leaf_count = 0
                for i in range(N):
                    node = tree.nodes[assignment[i]]
                    if node.left_child == 0:
                        leaf_count += 1
                        continue
                    if node.is_owner:
                        if features[node.feature_id][i] < node.threshold:
                            assignment[i] = node.left_child
                        else:
                            assignment[i] = node.right_child
                    else:
                        assignment[i] = -1
                print(1)
                bridge.send(bridge.current_iter_id, 'leader_assignment', assignment)
                print(2)
                follower_assignment = bridge.receive(bridge.current_iter_id, 'follower_assignment')
                print(3)
                assignment = np.maximum(assignment, follower_assignment)
                bridge.commit()
            for i in range(N):
                fx[i] += tree.nodes[assignment[i]].weight

        return self._loss.predict(fx)

    def _batch_predict_follower(self, bridge, example_ids, features):
        N = len(example_ids)
        for tree in self._proto.trees:
            assignment = np.zeros(N, dtype=np.int32)
            leaf_count = 0
            while leaf_count != N:
                bridge.start(bridge.new_iter_id())
                leaf_count = 0
                for i in range(N):
                    node = tree.nodes[assignment[i]]
                    if node.left_child == 0:
                        leaf_count += 1
                        continue
                    if node.is_owner:
                        if features[node.feature_id][i] < node.threshold:
                            assignment[i] = node.left_child
                        else:
                            assignment[i] = node.right_child
                    else:
                        assignment[i] = -1
                print(1)
                bridge.send(bridge.current_iter_id, 'follower_assignment', assignment)
                print(2)
                leader_assignment = bridge.receive(bridge.current_iter_id, 'leader_assignment')
                print(3)
                assignment = np.maximum(assignment, leader_assignment)
                bridge.commit()


    def fit(self, bridge, example_ids, features, labels=None):
        num_examples = len(example_ids)

        # make key pair
        bridge.start(bridge.new_iter_id())
        if bridge.role == 'leader':
            public_key, private_key = paillier.PaillierKeypair.generate_keypair(KEY_NBITS)
            _send_public_key(bridge, public_key)
        else:
            public_key = _receive_public_key(bridge)
        bridge.commit()

        # sort feature columns
        ptrs = {fid: np.argsort(x) for fid, x in features.items()}
        sorted_features = {fid: x[ptrs[fid]] for fid, x in features.items()}

        # initial f(x)
        if bridge.role == 'leader':
            sum_fx = np.zeros(num_examples, dtype=BST_TYPE)
            for t in range(self._params.num_round):
                tree, fx = self._fit_one_round_leader(
                    public_key, private_key,
                    bridge, sum_fx, example_ids, features, ptrs, sorted_features, labels)
                self._proto.trees.append(tree)
                sum_fx += fx
        else:
            for t in range(self._params.num_round):
                tree = self._fit_one_round_follower(
                    public_key, bridge, example_ids, features, ptrs, sorted_features)
                self._proto.trees.append(tree)

    def _fit_one_round_leader(self, public_key, private_key, bridge, sum_fx, example_ids, features, ptrs, sorted_features, labels):
        num_examples = len(example_ids)

        # initialize
        tree = tree_pb2.RegressionTreeProto()
        tree.nodes.append(
            tree_pb2.RegressionTreeNodeProto(
                node_id=len(tree.nodes), weight=0))
        fx = np.zeros(num_examples, dtype=BST_TYPE)
        leaf_nodes = [tree.nodes[0]]
        assignment = np.zeros(num_examples, dtype=np.int64)
        num_bins = int(1.0/self._params.sketch_eps)
        bin_size = (num_examples + num_bins - 1) // num_bins

        # fit
        for d in range(self._params.max_depth):
            # start iteration
            bridge.start(bridge.new_iter_id())

            # compute grad and hess
            x = sum_fx + fx
            pred = self._loss.predict(x)
            grad = self._loss.gradient(x, pred, labels)
            hess = self._loss.hessian(x, pred, labels)
            print('iter %d metrics: %s'%(d, self._loss.metrics(pred, labels)))
            _encrypt_and_send_numbers(bridge, 'grad', public_key, grad)
            _encrypt_and_send_numbers(bridge, 'hess', public_key, hess)

            # compute leaf purity
            example_per_node = np.bincount(assignment)
            pos_per_node = np.bincount(assignment, weights=labels)
            print('leaf purity:', pos_per_node/example_per_node)

            # compute histogram
            grad_hists = _compute_histogram(
                grad, ptrs, leaf_nodes, assignment, num_bins, bin_size)
            hess_hists = _compute_histogram(
                hess, ptrs, leaf_nodes, assignment, num_bins, bin_size)
            follower_grad_hists = _receive_and_decrypt_histogram(
                bridge, 'grad_hists', public_key, private_key, num_bins, len(leaf_nodes))
            follower_hess_hists = _receive_and_decrypt_histogram(
                bridge, 'hess_hists', public_key, private_key, num_bins, len(leaf_nodes))

            # find split
            new_leaf_nodes = []
            for nid, node in enumerate(leaf_nodes):
                node_grad_hists = {fid: hist[nid] for fid, hist in grad_hists.items()}
                node_grad_hists.update({fid: hist[nid] for fid, hist in follower_grad_hists.items()})
                node_hess_hists = {fid: hist[nid] for fid, hist in hess_hists.items()}
                node_hess_hists.update({fid: hist[nid] for fid, hist in follower_hess_hists.items()})
                max_gain, max_fid, split_point, left_weight, right_weight = _find_split(
                    node_grad_hists, node_hess_hists, self._params.lam)

                if max_fid in grad_hists:
                    left_child, right_child = _split_node(tree, node, bin_size, max_fid, sorted_features, split_point, left_weight, right_weight)
                    bridge.send_proto(
                        bridge.current_iter_id, 'split_node_%d'%nid,
                        tree_pb2.SplitInfo(owner_id=0))
                else:
                    left_child, right_child = _split_node(tree, node, bin_size, left_weight=left_weight, right_weight=right_weight)
                    bridge.send_proto(
                        bridge.current_iter_id, 'split_node_%d'%nid,
                        tree_pb2.SplitInfo(
                            owner_id=1, feature_id=max_fid, split_point=split_point))
                new_leaf_nodes.extend([left_child, right_child])

            # update assignment and fx
            follower_assignment = bridge.receive(bridge.current_iter_id, 'follower_assignment')
            for i in range(num_examples):
                lid = assignment[i]
                node = leaf_nodes[lid]
                if node.is_owner:
                    if features[node.feature_id][i] < node.threshold:
                        assignment[i] = lid * 2
                        fx[i] = new_leaf_nodes[lid*2].weight
                    else:
                        assignment[i] = lid * 2 + 1
                        fx[i] = new_leaf_nodes[lid*2 + 1].weight
                else:
                    assignment[i] = follower_assignment[i]
                    assert assignment[i] >= 0 and assignment[i] < len(new_leaf_nodes)
                    fx[i] = new_leaf_nodes[assignment[i]].weight

            if d + 1 < self._params.max_depth:
                bridge.send(bridge.current_iter_id, 'leader_assignment', assignment)

            leaf_nodes = new_leaf_nodes

            # commit iteration
            bridge.commit()

        self._proto.trees.append(tree)

        return tree, fx


    def _fit_one_round_follower(self, public_key, bridge, example_ids, features, ptrs, sorted_features):
        num_examples = len(example_ids)

        # initialize
        tree = tree_pb2.RegressionTreeProto()
        tree.nodes.append(
            tree_pb2.RegressionTreeNodeProto(
                node_id=len(tree.nodes), weight=0))
        leaf_nodes = [tree.nodes[0]]
        assignment = np.zeros(num_examples, dtype=np.int64)
        num_bins = int(1.0/self._params.sketch_eps)
        bin_size = (num_examples + num_bins - 1) // num_bins

        # fit
        for d in range(self._params.max_depth):
            # start iteration
            bridge.start(bridge.new_iter_id())

            # compute grad and hess
            grad = _receive_encrypted_numbers(bridge, 'grad', public_key)
            hess = _receive_encrypted_numbers(bridge, 'hess', public_key)

            # compute leaf purity

            # compute histogram
            zero = public_key.encrypt(0.0, PRECISION)
            grad_hists = _compute_histogram(
                grad, ptrs, leaf_nodes, assignment, num_bins, bin_size, zero=zero)
            hess_hists = _compute_histogram(
                hess, ptrs, leaf_nodes, assignment, num_bins, bin_size, zero=zero)
            _send_histograms(bridge, 'grad_hists', grad_hists)
            _send_histograms(bridge, 'hess_hists', hess_hists)

            # find split
            new_leaf_nodes = []
            for nid, node in enumerate(leaf_nodes):
                msg = tree_pb2.SplitInfo()
                bridge.receive_proto(bridge.current_iter_id, 'split_node_%d'%nid).Unpack(msg)
                if msg.owner_id == 0:
                    left_child, right_child = _split_node(tree, node, bin_size)
                else:
                    left_child, right_child = _split_node(tree, node, bin_size, msg.feature_id, sorted_features, msg.split_point)
                new_leaf_nodes.extend([left_child, right_child])

            # update assignment and fx
            for i in range(num_examples):
                lid = assignment[i]
                node = leaf_nodes[lid]
                if node.is_owner:
                    if features[node.feature_id][i] < node.threshold:
                        assignment[i] = lid * 2
                    else:
                        assignment[i] = lid * 2 + 1
                else:
                    assignment[i] = -1
            bridge.send(bridge.current_iter_id, 'follower_assignment', assignment)

            if d + 1 < self._params.max_depth:
                assignment[:] = bridge.receive(bridge.current_iter_id, 'leader_assignment')

            leaf_nodes = new_leaf_nodes

            # commit iteration
            bridge.commit()

        self._proto.trees.append(tree)

        return tree

