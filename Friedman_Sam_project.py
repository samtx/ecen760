# Sam Friedman
# 10/9/2018
# HW 2
# ECEN 760

from __future__ import print_function

import sys

import numpy as np

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

        self.build_parent_and_child_sets()
        self.build_cpds()

        self.root_nodes = set()
        self.leaf_nodes = set()
        self.find_root_and_leaf_nodes()

    def build_cpds(self):
        """
        Build dict of blank cpds based on number of parents per node
        """
        for node in self.nodes:
            num_parents = len(self.get_parents(node))
            self.cpd[node] = CPD(num_parents+1)

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

            data = x.copy()
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

    def build_lambda_and_pi_values(self):
        for node in self.nodes:
            # initialize all lambda and pi values to 1
            self.lambda_values.update({node:[1.0, 1.0]})
            self.pi_values.update({node:[1.0, 1.0]})

    def update_lambda_value(self, x, val_x, lambda_val):
        self.lambda_values[x][val_x] = lambda_val

    def update_pi_value(self, x, val_x, pi_val):
        self.pi_values[x][val_x] = pi_val

    def find_root_and_leaf_nodes(self):
        for node in self.nodes:
            if node not in self.parents:
                self.root_nodes.add(node)
            if node not in self.children:
                self.leaf_nodes.add(node)

    def Pearl(self, x=[], e=[]):
        """
        Implement Pearl's message passing algorithm for finding the conditional
        probability of X=x given E=e
        :param x: a tuple with (node, value)
        :param e: a list of tuples [(node, value), ..., (node,value)]
        :return:
        """
        # Initialize network
        self.observed = set()  # observed nodes and values

        # Initialize lambda and pi values for all nodes and node values
        self.lambda_values = {}
        self.pi_values = {}
        self.build_lambda_and_pi_values()

        # loop over root nodes
        for node in self.root_nodes:
            for val in [0,1]:
                prob = self.query_cpd([(node, val)],[])
                #print(node, val, prob)
                self.pi_values[node][val] = prob
            for W in self.get_children(node):
                self.send_pi_msg(node, W)

    def update_network(self, node, val):
        """
        observed set = {(E,e)... (E,e)}
        """
        # add observed node and value to observed set
        self.observed.add((node,val))

        for v in [0,1]:
            if v == val:
                self.update_lambda_value(node, v, 1.)
                self.update_pi_value(node, v, 1.)
            else:
                self.update_lambda_value(node, v, 0.)
                self.update_pi_value(node, v, 0.)

        node_set = self.get_parents(node) - self.observed
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
        W = self.get_parents(Y) - X

        for val in [0,1]:
            pass


    def send_pi_msg(self, node, Y):
        pass

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

    def is_active(self, a, b, Z):
        """
        Verify if there exists an active trail from node a to b given evidence set Z
        """
        # Find set Y of all nodes that are d_separated from a
        Y = self.d_separated(a, Z)

        # If b is in Y then there is no active trail
        return not (b in Y)

    def d_separated(self, X, Z):
        """
        Find the set Y of all d-separated nodes such that

                d-sep_G(X indep Y | Z) is True

        Uses the Reachable() algorithm 3.1 in PGM book
        """
        # Find all reachable nodes from node X given Z
        R = self.reachable(X, Z)

        # Return nodes in G that are not in R and not in Z
        dsep_nodes = self.nodes - R - Z

        return dsep_nodes

    def reachable(self, X, Z):
        """ 
        Find all nodes that are reachable via an active trail from the given
        node and evidence set.

        Based on Algorithm 3.1, page 75, in PGM book

        inputs:
            X: starting node
            Z: set of observations
        outputs:
            R: set of nodes reachable via active trail
        """
        # Phase 1: insert all ancestors of Z into A
        L = Z.copy()  # set L to be the set of observations
        A = set()  # set A to be the empty set
        while len(L) != 0:
            Y = L.pop()  # remove an item from the set
            if Y not in A:
                L = L.union(self.get_parents(Y))  # Y's parents need to be visited
            A = A.union(Y)  # Y is ancestor of evidence

        # Phase 2: traverse active trails starting from node
        L = {(X, 'up')}
        V = set()  # (node, direction) marked as visited
        R = set()  # Nodes reachable via active trail
        while len(L) != 0:
            # select some (Y, dr) from L
            (Y, dr) = L.pop()
            if (Y, dr) not in V:
                if Y not in Z:
                    R = R | {Y}  # Y is reachable
                V = V | {(Y, dr)}  # mark (Y, dr) as visited
                if dr == 'up' and Y not in Z:
                    # trail up through Y active if Y not in Z
                    for z in self.get_parents(Y):
                        L = L.union({(z, 'up')})  # Y's parents to be visited from bottom
                    for z in self.get_children(Y):
                        L = L.union({(z, 'down')})  # Y's children to be visited from top
                elif dr == 'down':
                    # trails down through Y
                    if Y not in Z:
                        # downward trails to Y's children are active
                        for z in self.get_children(Y):
                            L = L.union({(z, 'down')})  # Y's children to be visited from top
                    if Y in A:
                        # v-structure trails are active
                        for z in self.get_parents(Y):
                            L = L.union({(z, 'up')})  # Y's parents to be visited from bottom
        return R


class CPD(object):
    """
    Conditional probability distribution table for binary random variables
    """
    def __init__(self, ndim):
        size_array = [2]
        if ndim > 1:
            for i in range(ndim-1):
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
        prob_idx = list(np.zeros([len(data_list)], dtype=int))  # index of cpd to change
        for data in data_list:
            idx = self.var_index.index(data[0])
            prob_idx[idx] = data[1]
        alt_prob_idx = prob_idx.copy()
        print(alt_prob_idx)
        alt_prob_idx[0] = 1 - alt_prob_idx[0]  # flip the outcome
        prob_idx = tuple(prob_idx)
        alt_prob_idx = tuple(alt_prob_idx)


        # update cpd with probability value
        prob_value = data_list[0][2]
        # cpd = self.cpd
        self.cpd[prob_idx] = prob_value
        self.cpd[alt_prob_idx] = np.round(1 - prob_value, 6)

        print('\n')
        print(data_list[0][0], '\n', self.cpd)
        print('\n')


def read_file(fname):
    """
    Read problem file, generate graph and questions
    """
    edges = set()
    queries = []
    cpds = []
    V, M, Q, R = 0, 0, 0, 0
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
                # Q: number of queries

            # Edges
            elif all([x.isalpha() for x in line]) and len(edges) < M:
                u, v = line
                edges = edges | {(u, v)}

            # Conditional Probability Distributions
            elif '=' in line:
                # build data list
                data_list = []
                prob = float(line[-1])
                data_list.append((line[0][0],int(line[0][1]),prob))
                for x in line[1:]:
                    if x == '|':
                        continue
                    if x == '=':
                        break
                    if x[0].isalpha() and x[1].isdigit():
                        data_list.append((x[0],int(x[1])))
                cpds.append(data_list)

            # Queries
            elif line[1] == '|' and len(queries) < Q:
                y = line[0]  # source node Y
                z = {x for x in line[2:]}  # evidence set Z
                queries.append((y, z))

            # Create Graph object
            if (len(edges) == M) and (len(queries) == Q) and (len(cpds) == R):
                G = Graph(edges=edges)
                for cpd in cpds:
                    print(cpd)
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

    return G, queries


if __name__ == "__main__":

    if len(sys.argv) > 1:
        # read filename as argument
        fname = sys.argv[1]
        G, queries = read_file(fname)
        # Run queries from file
        print(G.edges)
        for cpd in G.cpd:
            print(cpd)

        # C1 | A0, B1 = 0.5
        # C1 | A1, B1 = 0.9
        # D1 | C0 = 0.8
        # D1 | C1 =
        #
        query = [[('G',0)],[('D',1)]]
        print(G.query_cpd(query[0],query[1]))
        print(G.query_cpd("G0 | D1"))
        print(G.query_cpd("C1 | A1 B0"))

        print(G.root_nodes)
        print(G.leaf_nodes)

        G.Pearl()

        for key, value in list(G.pi_values.items()):
            print(key, value)

        for query in queries:
            X, Z = query
            # Evaluate query
            dsep_nodes = G.d_separated(X, Z)
            # print results to stdout
            if not dsep_nodes:
                out_str = 'None'
            else:
                out_str = ""
                for x in dsep_nodes:
                    out_str += str(x) + " "
            print(out_str)

    else:
        # Build Graph from homework 1, problem 3
        edges = {('A', 'D'), ('B', 'D'), ('D', 'G'), ('D', 'H'), ('G', 'K'), ('H', 'K'),
                 ('H', 'E'), ('C', 'E'), ('E', 'I'), ('F', 'I'), ('F', 'J'), ('I', 'L'),
                 ('J', 'M')}
        G = Graph(edges=edges)

        # Check answers to HW 1, Problem 3
        print('Check answers from HW 1, Problem 3:')

        # parts 3(a)-(e)
        queries = {
            'a': ('A', 'J', {'G', 'L'}),
            'b': ('A', 'C', {'L'}),
            'c': ('G', 'L', {'D'}),
            'd': ('G', 'L', {'D', 'K', 'M'}),
            'e': ('B', 'F', {'C', 'G', 'L'})
        }
        for i in ['a', 'b', 'c', 'd', 'e']:
            q = queries[i]
            out_str = '(3{}) Active trail from {} to {} given {}? {}'.format(
                i,
                q[0],
                q[1],
                q[2],
                G.is_active(q[0], q[1], q[2]))
            print(out_str)

        # parts 3(f)-(g)
        queries.update({
            'f': ('A', {'K', 'E'}),
            'g': ('B', {'L'})
        })
        for i in ['f', 'g']:
            q = queries[i]
            out_str = '(3{}) Nodes d-separated from {} given {} = {}'.format(
                i,
                q[0],
                q[1],
                G.d_separated(q[0], q[1]))
            print(out_str)