# Sam Friedman
# 11/29/2018
# Project
# ECEN 760

from __future__ import print_function

import numpy as np   # Needs to be downloaded
import itertools     # standard library
from collections import OrderedDict  # standard library
import sys  # standard library
import re   # standard library
import pprint


DEBUG = False

class Graph(object):
    """
    Graph object that has sets of nodes and edges
    """

    def __init__(self, edges=set(), nodes=set()):
        self.parents = {}
        self.children = {}
        self.nodes = nodes
        if len(nodes) > 0:
            self.size = len(nodes)
        else:
            self.size = 0
        self.edges = edges
        self.cpd = {}  # conditional probability distributions
        self.p = {}    # marginal probability distributions

        self.build_parent_and_child_sets()
        self.build_cpds()
        self.build_marginals()

        self.root_nodes = set()
        self.leaf_nodes = set()
        self.find_root_and_leaf_nodes()

    def build_marginals(self):
        """
        Build dict of blank arrays for each node
        """
        for node in self.nodes:
            self.p.update({
                node: np.zeros(2)  # binary variables
            })

    # def update_marginal(self, node, value, prob):
    #     """
    #     Set the marginal probability for the given node and value
    #     """
    #     self.p[node][value] = prob

    def build_cpds(self):
        """
        Build dict of blank cpds based on number of parents per node
        """
        for node in self.nodes:
            num_parents = len(self.get_parents(node))
            self.cpd[node] = CPD(num_parents + 1)

    def update_cpd(self, data_list):
        node = data_list[0][0]
        self.cpd[node].update(data_list)

    def parse_cpd_query(self, query_string):
        """
        example: 'X1 | Y0 Z1'
        """
        x = []
        e = []
        line = query_string.split('|')
        x_line = line[0].split()
        e_line = line[1].split()
        for item in x_line:
            if item[0].isalpha() and item[1].isdigit():
                x.append((item[0], int(item[1])))
        for item in e_line:
            if item[0].isalpha() and item[1].isdigit():
                e.append((item[0], int(item[1])))
        return x, e

    def query_cpd(self, x, e=[]):
        """
        x : list of nodes with values... [(node,val),(node,val)]
        e : same as x
        return P(x|e)
        """
        if isinstance(x, str):
            x, e = self.parse_cpd_query(x)

        if len(x) == 1:

            prob_idx = [0 for i in self.cpd[x[0][0]].var_index]

            data = x[:]
            for y in e:
                data.append(y)
            for y in data:
                var, val = y
                var_idx = self.cpd[x[0][0]].var_index.index(var)
                prob_idx[var_idx] = val
            prob_idx = tuple(prob_idx)
            return self.cpd[x[0][0]].cpd[prob_idx]
        else:
            print("Can't query joint conditional prob yet!")

    def build_lambda_and_pi_values_and_msg(self):
        for node in self.nodes:
            # initialize all lambda and pi values and msg to 1
            self.lambda_values.update({node: [1.0, 1.0]})
            self.pi_values.update({node: [1.0, 1.0]})

            # lambda msgs are sent from child to parent for each parent value
            # lambda_msg = {
            #   child1 : {parent1:[parent1_val0, parent1_val1], parent2:[parent2_val0, parent2_val1]},
            #   child2 : {parent1:[parent1_val0,...
            # }
            self.lambda_msg.update({node: {}})
            for parent in self.get_parents(node):
                # init lambda msg to 1
                # lambda_{child}(parent node value)
                if DEBUG:
                    print('Node={}  Parent={}'.format(node, parent))
                self.lambda_msg[node].update({parent: [1.0, 1.0]})

            # pi msgs are sent from parent to child for each parent value
            # pi_msg = {
            #   parent1 : {child1: [parent1_val0, parent1_val1], child2:[parent1_val0, parent1_val1]},
            #   parent2 : {child1: [parent2_val0, parent2_val1],...}
            # }
            self.pi_msg.update({node: {}})
            for child in self.get_children(node):
                # init pi msg to 1
                if DEBUG:
                    print('Node={}  Child={}'.format(node, child))
                self.pi_msg[node].update({child: [1.0, 1.0]})

    def update_lambda_value(self, x, val_x, lambda_val):
        self.lambda_values[x][val_x] = lambda_val

    def update_pi_value(self, x, val_x, pi_val):
        self.pi_values[x][val_x] = pi_val

    def update_lambda_msg(self, X, U, val_x, lambda_msg_val):
        """
        lambda_{U} (x = x_val)
            X : parent
            U : child

        (X=x)---->(U)

        lambda msg are sent from child to parent node for parent value
        """
        if DEBUG:
            print('Update Lambda Msg')
            print('U={}  X={}  val_x={}'.format(U, X, val_x))
        self.lambda_msg[U][X][val_x] = lambda_msg_val

    def update_pi_msg(self, X, W, val_w, pi_msg_val):
        """
        pi_{x} (w = w_val)
            W : parent
            X : child

        (W=w)--->(X)

        pi msg are sent from parent to child for parent value
        """
        if DEBUG:
            print('Update Pi Msg')
            print('W={}  X={}  val_w={}'.format(W, X, val_w))
        self.pi_msg[W][X][val_w] = pi_msg_val

    def get_lambda_value(self, x, val_x):
        if DEBUG:
            print('Get Lambda Value')
            print('x={}, val_x={}'.format(x, val_x))
        return self.lambda_values[x][val_x]

    def get_pi_value(self, x, val_x):
        return self.pi_values[x][val_x]

    def get_lambda_msg(self, x, child, val_x):
        """
        lambda_{child Y} (node x = val_x)
        """
        return self.lambda_msg[child][x][val_x]

    def get_pi_msg(self, x, child, val_x):
        return self.pi_msg[x][child][val_x]

    def find_root_and_leaf_nodes(self):
        for node in self.nodes:
            if node not in self.parents:
                self.root_nodes.add(node)
            if node not in self.children:
                self.leaf_nodes.add(node)

    def infer(self, x=None, e=None):
        """
        Infer the probability of X=x given E=e
        """
        if len(x) == 1:
            # no joint probability
            prob = self.Pearl(x[0], e)
        else:
            # joint probability
            print('Joint probability not implemented yet!')
            prob = 0.0
        return prob

    def Pearl(self, X, E=[]):
        """
        Implement Pearl's message passing algorithm for finding the conditional
        probability of X=x given E=e
        :param x: a tuple with (node, value)
        :param e: a list of tuples [(node, value), ..., (node,value)]
        :return:
        """

        self.initialize_network()

        if DEBUG:
            print('Lambda Values')
            pprint.pprint(self.lambda_values)

            print('Pi Values')
            pprint.pprint(self.pi_values)

            print('Lambda Messages')
            pprint.pprint(self.lambda_msg)

            print('Pi Msg')
            pprint.pprint(self.pi_msg)

        for E_i in E:
            # E_i = (E_node, E_val)
            self.update_network(E_i[0], E_i[1])

        sum_x = 0.0
        for val_x in [0, 1]:
            sum_x += self.get_lambda_value(X[0], val_x) * self.get_pi_value(X[0], val_x)
        p = (1 / sum_x) * self.get_lambda_value(X[0], X[1]) * self.get_pi_value(X[0], X[1])

        return p

    def initialize_network(self):
        # Initialize network
        self.observed = {}  # observed nodes and values

        # Initialize lambda and pi values for all nodes and node values
        self.lambda_values = {}
        self.pi_values = {}
        self.lambda_msg = {}
        self.pi_msg = {}
        self.build_lambda_and_pi_values_and_msg()

        if DEBUG:
            print('Lambda Values')
            pprint.pprint(self.lambda_values)

            print('Pi Values')
            pprint.pprint(self.pi_values)

            print('Lambda Messages')
            pprint.pprint(self.lambda_msg)

            print('Pi Msg')
            pprint.pprint(self.pi_msg)

        # loop over root nodes
        for node in self.root_nodes:
            for val in [0, 1]:
                prob = self.query_cpd([(node, val)], [])
                self.update_pi_value(node, val, prob)
            for W in self.get_children(node):
                self.send_pi_msg(node, W)

    def update_network(self, node, val):
        """
        observed set = {(E,e)... (E,e)}
        """
        # add observed node and value to observed set
        self.observed.update({node: val})

        for v in [0, 1]:
            if v == val:
                self.update_lambda_value(node, v, 1.)
                self.update_pi_value(node, v, 1.)
            else:
                self.update_lambda_value(node, v, 0.)
                self.update_pi_value(node, v, 0.)

        node_set = self.get_parents(node) - set(self.observed.keys())
        for Z in node_set:
            self.send_lambda_msg(node, Z)
        for Y in self.get_children(node):
            self.send_pi_msg(node, Y)

    def send_lambda_msg(self, Y, X):
        """
        Y : child
        X : parent
        """

        # get other parents of Y besides X
        W_set = self.get_parents(Y) - {X}
        k = len(W_set)  # size of W set

        P_hat = [0., 0.]

        for val_x in [0, 1]:

            # Sum over values of Y
            sum_y = 0.

            for val_y in [0,1]:

                # Sum over values of W's
                sum_w = 0.

                # Create list of conditional nodes and values
                cond_values = [(X, val_x)]
                for w_node in W_set:
                    cond_values.append([w_node, 0])

                # generate full factorial of all possible values for w
                w_full_factorial = itertools.product([0, 1], repeat=k)

                for w_list in w_full_factorial:

                    # set values of W nodes
                    w_i_idx = 1  # starting index for w_i
                    for w_i_val in w_list:
                        cond_values[w_i_idx][1] = w_i_val
                        w_i_idx += 1

                    prob_y_given_x = self.query_cpd([(Y, val_y)], cond_values)

                    # get product of pi msg of Y
                    pi_msgs = 1.
                    for w_item in cond_values[1:]:
                        pi_msgs *= self.get_pi_msg(w_item[0], Y, w_item[1])

                    sum_w += prob_y_given_x * pi_msgs

                sum_y += sum_w * self.get_lambda_value(Y, val_y)

            # Update lambda message with calculated sum
            self.update_lambda_msg(X, Y, val_x, sum_y)

            # Set P hat
            P_hat[val_x] = self.get_lambda_value(X, val_x) * self.get_pi_value(X, val_x)

        alpha = np.sum(P_hat)

        # calculate P(x|e) with alpha
        # ....

        # Send lambda message to parents of X
        Z_set = self.get_parents(X) - set(self.observed.keys())
        for z_node in Z_set:
            self.send_lambda_msg(X, z_node)

        # Send pi message to children of X
        U_set = self.get_children(X) - {Y}
        for u_node in U_set:
            self.send_pi_msg(X, u_node)



    def send_pi_msg(self, Z, X):
        """
        Send pi-message from parent 'Z' to child 'X'
        :param Z:
        :param X:
        :return:
        """

        # For each value of node Z
        for val_z in [0, 1]:

            # calculate pi message to child X from Z
            U_set = self.get_children(Z) - {X}

            # calculate product of lambda messages from other children of Z besides X
            lambda_msgs = 1.
            for u_node in U_set:
                lambda_msgs *= self.get_lambda_msg(Z, u_node, val_z)

            pi_msg_val = self.get_pi_value(Z, val_z) * lambda_msgs
            # update pi msg
            if DEBUG:
                pprint.pprint(self.pi_msg)
                print('X={}  Z={}  val_z={:d}  pi_msg_val={:.4f}'.format(X,Z,val_z,pi_msg_val))
            self.update_pi_msg(X, Z, val_z, pi_msg_val)

        if X not in self.observed:
            for val_x in [0, 1]:
                # Z1 ... Zk are the parents of X
                sum_z = 0.0

                Z_set = self.get_parents(X)

                k = len(Z_set)
                # generate full factorial of all possible values for Z_i=z_i
                z_full_factorial = itertools.product([0, 1], repeat=k)

                # Create list of conditional nodes and values
                cond_values = []
                for z_node in Z_set:
                    cond_values.append([z_node, 0])

                for z_list in z_full_factorial:

                    # set values of Z nodes
                    z_i_idx = 0  # starting index for z_i
                    for z_i_val in z_list:
                        cond_values[z_i_idx][1] = z_i_val
                        z_i_idx += 1

                    prob_x_given_z = self.query_cpd([(X, val_x)], cond_values)

                    # get product of pi msg of X
                    pi_msgs = 1.
                    for z_item in cond_values:
                        pi_msgs *= self.get_pi_msg(z_item[0], X, z_item[1])

                    sum_z += prob_x_given_z * pi_msgs

                self.update_pi_value(X, val_x, sum_z)

            # alpha = sum_{x}(P_tilde(x))

            for Y in self.get_children(X):
                self.send_pi_msg(X, Y)

        x_lambda_vals = []
        for val_x in [0, 1]:
            x_lambda_vals.append(self.get_lambda_value(X, val_x))

        if any([i != 1.0 for i in x_lambda_vals]):
            node_set = self.get_parents(X) - {Z}
            for W in node_set:
                if W in self.observed:
                    # skip W if already observed
                    continue
                self.send_lambda_msg(X, W)

    def add_node(self, node):
        """
        Add node to graph
        """
        if node not in self.nodes:
            self.size += 1
            self.nodes.add((node))
        else:
            pass

    def build_parent_and_child_sets(self):
        """
        Loop through set of edges and
            - add nodes to Graph object
            - create Graph.parents, Graph.children dictionaries
        """
        for edge in self.edges:
            u, v = edge
            self.add_node(u)
            self.add_node(v)

            if v not in self.parents:
                self.parents.update({v: set(u)})
            else:
                self.parents[v].add(u)

            if u not in self.children:
                self.children.update({u: set(v)})
            else:
                self.children[u].add(v)

    def get_parents(self, node):
        """
        Return parents of node
        """
        if node in self.parents:
            return self.parents[node]
        else:
            return set()

    def get_children(self, node):
        """
        Return children of node
        """
        if node in self.children:
            return self.children[node]
        else:
            return set()




class CPD(object):
    """
    Conditional probability distribution table for binary random variables
    """

    def __init__(self, ndim):
        size_array = [2]
        if ndim > 1:
            for i in range(ndim - 1):
                size_array.append(2)
        self.cpd = np.zeros(size_array)
        self.var_index = []

    def build_index(self, var_list):
        for var_name in var_list:
            self.var_index.append(var_name)

    def update(self, data_list):
        """
        :param data_list:  [(variable, value, probability),(givenVar, givenVal),...,(givenVar, givenVal)]
        :return:
        """
        # build var name index
        if not self.var_index:
            var_list = [data[0] for data in data_list]
            self.build_index(var_list)

        # get index of cpd to update
        # prob_idx = list(np.zeros([len(data_list)], dtype=int))  # index of cpd to change
        prob_idx = [0]*len(data_list) # index of cpd to change
        for data in data_list:
            idx = self.var_index.index(data[0])
            prob_idx[idx] = data[1]
        alt_prob_idx = prob_idx[:]
        # print(alt_prob_idx)
        alt_prob_idx[0] = 1 - alt_prob_idx[0]  # flip the outcome
        prob_idx = tuple(prob_idx)
        alt_prob_idx = tuple(alt_prob_idx)

        # update cpd with probability value
        prob_value = data_list[0][2]
        # cpd = self.cpd
        self.cpd[prob_idx] = prob_value
        self.cpd[alt_prob_idx] = np.round(1 - prob_value, 6)

        # print('\n')
        # print(data_list[0][0], '\n', self.cpd)
        # print('\n')


def read_problem_query(query_str):
    q_num_regex = "^\(\w+\)"
    query_regex = "P\([,\s\w]+\)|P\([,\s\w]+\|[,\s\w]+\)"
    # find question number
    q_num = re.search(q_num_regex, query_str)
    problem_query = {
        'question_number': q_num.group(0),
        'query_list': []
    }
    for match in re.finditer(query_regex,query_str):
        query = match.group(0)  # get individual query
        problem_query['query_list'].append(query)  # add to query list
    return problem_query

def parse_query(one_query_str):
    """
    one_query_str = 'P(A1, B0 | D0, F1)'
                  = 'P(A1)'
                  = 'P(A1|D1,F0)'
    """

    x = [] # nodes to query
    e = [] # evidence nodes
    query = one_query_str.lstrip('P(').rstrip(')') # remove P( and ) from string
    query = query.split('|')  # split by query nodes and evidence nodes
    x_tmp = query[0]
    e_tmp = query[1]
    return x, e


def read_file(fname):
    """
    Read problem file, generate graph and questions
    """
    edges = set()
    queries = []
    cpds = []
    V, M, Q, R = 0, 0, 0, 0
    query_regex = "P\([,\s\w]+\)|P\([,\s\w]+\|[,\s\w]+\)"
    query_check = re.compile(query_regex)
    with open(fname) as f:
        for raw_line in f:
            # split the line into components separated by whitespace
            line = raw_line.split()
            # skip '#' as comment
            if line[0][0] == '#':
                continue

            # New Graph description
            elif all([x.isdigit() for x in line]):
                edges = set()
                queries = []
                cpds = []
                V, M, R, Q = [int(x) for x in line]
                # V: number of nodes
                # M: number of edges
                # R: number of CPDs
                # Q: number of problem sets

            # Edges
            elif all([x.isalpha() for x in line]) and len(edges) < M:
                u, v = line
                edges = edges | {(u, v)}

            # Conditional Probability Distributions
            elif '=' in line:
                # build data list
                data_list = []
                prob = float(line[-1])
                data_list.append((line[0][0], int(line[0][1]), prob))
                for x in line[1:]:
                    if x == '|':
                        continue
                    if x == '=':
                        break
                    if x[0].isalpha() and x[1].isdigit():
                        data_list.append((x[0], int(x[1])))
                cpds.append(data_list)

            # Queries
            # use more complicated regex if there is time
            # regex: P\([,\s\w]+\)|P\([,\s\w]+\|[,\s\w]+\)
            #regex = 'P\([,\s\w]+\)|P\([,\s\w]+\|[,\s\w]+\)'

            # elif any([query_check.match(item) for item in line]) and (len(queries) < Q):
            #     print('its a query!')
            #     prob_dict = read_problem_query(line)
            #     queries.append(prob_dict)

    # Create Graph object
    if (len(edges) == M)  and (len(cpds) == R):
        G = Graph(edges=edges)
        print('make graph')
        for cpd in cpds:
            # print(cpd)
            G.update_cpd(cpd)

        # Validate graph
        err = False
        if len(G.nodes) != V:
            print('Not the correct number of nodes')
            err = True
        if len(G.edges) != M:
            print('Not the correct number of edges')
            err = True
        if err:
            return
    if DEBUG:
        print(len(edges),M, len(queries),Q,len(cpds), R)

    return G, queries


if __name__ == "__main__":

    if len(sys.argv) > 1:
        # read filename as argument
        fname = sys.argv[1]
        G, _ = read_file(fname)
        # Run queries from file
        # print(G.edges)
        # print(G.cpd)
        # for node, cpd in iter(G.cpd.items()):
        #     print(node, cpd.cpd)
        #
        queries = [
            [[('A', 1)], [('B', 0)]],
            [[('A', 1)], [('D', 0)]],
            [[('B', 1)], [('A', 1)]],
            [[('B', 1)], [('C', 1)]],
            [[('A', 1)], [('B', 0)]],
            [[('C', 1)], []],
            [[('C', 1)], [('A', 1)]],
            [[('A', 1), ('D', 1)], [('F', 0), ('B', 1)]]
        ]

        for q in queries:
            p = G.infer(q[0], q[1])
            print(q, p)
        # G.Pearl()
        #

    else:

        # Create list of conditional nodes and values
        k = 4
        # cond_values = [['x', 0], ['w1', 0], ['w2', 0], ['w3', 0], ['w4', 0]]
        W_set = {'w1','w2','w3','w4'}
        cond_values = [['x',0]]
        for w_node in W_set:
            cond_values.append([w_node, 0])

        # generate full factorial of all possible values for w
        w_full_factorial = itertools.product([0, 1], repeat=k)

        for w_list in w_full_factorial:

            # set values of W nodes
            w_i_idx = 1  # starting index for w_i
            for w_i_val in w_list:
                # print(w_i_idx)
                cond_values[w_i_idx][1] = w_i_val
                w_i_idx += 1
            print(cond_values)
        #
        #
        # # Build Graph from homework 1, problem 3
        # edges = {('A', 'D'), ('B', 'D'), ('D', 'G'), ('D', 'H'), ('G', 'K'), ('H', 'K'),
        #          ('H', 'E'), ('C', 'E'), ('E', 'I'), ('F', 'I'), ('F', 'J'), ('I', 'L'),
        #          ('J', 'M')}
        # G = Graph(edges=edges)
        #
        # # Check answers to HW 1, Problem 3
        # print('Check answers from HW 1, Problem 3:')
        #
        # # parts 3(a)-(e)
        # queries = {
        #     'a': ('A', 'J', {'G', 'L'}),
        #     'b': ('A', 'C', {'L'}),
        #     'c': ('G', 'L', {'D'}),
        #     'd': ('G', 'L', {'D', 'K', 'M'}),
        #     'e': ('B', 'F', {'C', 'G', 'L'})
        # }
        # for i in ['a', 'b', 'c', 'd', 'e']:
        #     q = queries[i]
        #     out_str = '(3{}) Active trail from {} to {} given {}? {}'.format(
        #         i,
        #         q[0],
        #         q[1],
        #         q[2],
        #         G.is_active(q[0], q[1], q[2]))
        #     print(out_str)
        #
        # # parts 3(f)-(g)
        # queries.update({
        #     'f': ('A', {'K', 'E'}),
        #     'g': ('B', {'L'})
        # })
        # for i in ['f', 'g']:
        #     q = queries[i]
        #     out_str = '(3{}) Nodes d-separated from {} given {} = {}'.format(
        #         i,
        #         q[0],
        #         q[1],
        #         G.d_separated(q[0], q[1]))
        #     print(out_str)
