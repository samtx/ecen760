# Sam Friedman
# 12/2/2018
# Project
# ECEN 760

from __future__ import print_function

import numpy as np   # Numpy >= 1.15.4, needs to be downloaded
import itertools     # standard library
import sys           # standard library
import re            # standard library


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

        # Add flag for credal networks
        self.is_credal = False

    def build_cpds(self):
        """
        Build dict of blank CPDs based on number of parents per node
        """
        for node in self.nodes:
            num_parents = len(self.get_parents(node))
            self.cpd[node] = CPD(num_parents + 1)

    def update_cpd(self, data_list):
        """
        Update conditional probability distribution
        """
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
        Query Bayesian Network conditional probability distribution

        Inputs:
            x : tuple of node with value... (node,val)
            e : list of tuples of conditional nodes and values
                [(node1,value1),...(nodeN,valueN)]

        Output:
            P(x|e)
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
        """
        Build dictionaries of lambda and pi values and messages
        Initialize all values and messages to 1.0
        """
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
                self.lambda_msg[node].update({parent: [1.0, 1.0]})

            # pi msgs are sent from parent to child for each parent value
            # pi_msg = {
            #   parent1 : {child1: [parent1_val0, parent1_val1], child2:[parent1_val0, parent1_val1]},
            #   parent2 : {child1: [parent2_val0, parent2_val1],...}
            # }
            self.pi_msg.update({node: {}})
            for child in self.get_children(node):
                # init pi msg to 1
                self.pi_msg[node].update({child: [1.0, 1.0]})

    def update_lambda_value(self, X, val_x, lambda_val):
        """
        Update the lambda value for
            lambda(X = val_x) = lambda_val
        """
        self.lambda_values[X][val_x] = lambda_val

    def update_pi_value(self, X, val_x, pi_val):
        """
        Update the pi value for
            pi(X = val_x) = pi_val
        """
        self.pi_values[X][val_x] = pi_val

    def update_lambda_msg(self, X, U, val_x, lambda_msg_val):
        """
        lambda_{U} (x = x_val)
            X : parent
            U : child

        (X=x)---->(U)

        lambda msg are sent from child to parent node for parent value
        """
        self.lambda_msg[U][X][val_x] = lambda_msg_val

    def update_pi_msg(self, X, W, val_w, pi_msg_val):
        """
        pi_{x} (w = w_val)
            W : parent
            X : child

        (W=w)--->(X)

        pi msg are sent from parent to child for parent value
        """
        self.pi_msg[W][X][val_w] = pi_msg_val

    def get_lambda_value(self, X, val_x):
        """
        Return the lambda value for
            lambda(X = val_x)
        """
        return self.lambda_values[X][val_x]

    def get_pi_value(self, X, val_x):
        """
        Return the pi value for
            pi(X = val_x)
        """
        return self.pi_values[X][val_x]

    def get_lambda_msg(self, X, child, val_x):
        """
        Return the lambda message for
           lambda_{child Y}(X = val_x)
        """
        return self.lambda_msg[child][X][val_x]

    def get_pi_msg(self, x, child, val_x):
        return self.pi_msg[x][child][val_x]

    def find_root_and_leaf_nodes(self):
        for node in self.nodes:
            if node not in self.parents:
                self.root_nodes.add(node)
            if node not in self.children:
                self.leaf_nodes.add(node)

    def infer(self, X, E=None):
        """
        Infer the probability of X=x given E=e.

        X can be a joint distribution

        Inputs:
            X: a list of tuples with [(node, value), ..., (node,value)]
            E: same as X
        note: X can be a query string, e.g. "P(A1,C1|B1,D0)"

        Output:
            p: probability of (joint) event calculated from inference
        """
        if isinstance(X, str):
            X, E = parse_query(X)

        prob = 1.
        while X:
            Xi = X.pop(0) # remove first node
            prob *= self.Pearl(Xi, E)
            E.append(Xi) # add node to evidence
        return prob

    def Pearl(self, X, E=[]):
        """
        Implement Pearl's message passing algorithm for finding the conditional
        probability of X=x given E=e

        Inputs:
            X: a tuple with (node, value)
            E: a list of tuples with [(node, value), ..., (node,value)]
        note: X can be a query string, e.g. "P(A1|B1)"

        Output:
            p: probability of event calculated from inference
        """
        if isinstance(X, str):
            X, E = parse_query(X)

        self.initialize_network()

        for E_i in E:
            self.update_network(E_i[0], E_i[1])

        sum_x = 0.0
        for val_x in [0, 1]:
            lambda_val = self.get_lambda_value(X[0], val_x)
            pi_val = self.get_pi_value(X[0], val_x)
            sum_x += lambda_val * pi_val
        p = (1 / sum_x) * self.get_lambda_value(X[0], X[1]) * self.get_pi_value(X[0], X[1])

        return p

    def initialize_network(self):
        """
        Initialize network for Pearl's message passing algorithm
        """
        # Initialize network
        self.observed = {}  # observed nodes and values

        # Initialize lambda and pi values for all nodes and node values
        self.lambda_values = {}
        self.pi_values = {}
        self.lambda_msg = {}
        self.pi_msg = {}
        self.build_lambda_and_pi_values_and_msg()

        # loop over root nodes
        for node in self.root_nodes:
            for val in [0, 1]:
                prob = self.query_cpd([(node, val)], [])
                self.update_pi_value(node, val, prob)
            for W in self.get_children(node):
                self.send_pi_msg(node, W)

    def update_network(self, node, val):
        """
        Update the network for Pearl's message passing algorithm

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
        Send a Lambda message from child 'Y' to parent 'X'
        Y : child node
        X : parent node
        """
        # get other parents of Y besides X
        W_set = self.get_parents(Y) - {X}
        k = len(W_set)  # size of W set

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
                        pi_msg = self.get_pi_msg(w_item[0], Y, w_item[1])
                        pi_msgs *= pi_msg

                    sum_w += prob_y_given_x * pi_msgs

                sum_y += sum_w * self.get_lambda_value(Y, val_y)

            # Update lambda message with calculated sum
            self.update_lambda_msg(X, Y, val_x, sum_y)

            # Update lambda value with product of lambda messages
            lambda_val = 1.0
            for U in self.get_children(X):
                lambda_val *= self.get_lambda_msg(X, U, val_x)
            self.update_lambda_value(X, val_x, lambda_val)

        # Send lambda message to parents of X
        Z_set = self.get_parents(X)
        for z_node in Z_set:
            if z_node in self.observed:
                continue
            self.send_lambda_msg(X, z_node)

        # Send pi message to children of X
        U_set = self.get_children(X) - {Y}
        for u_node in U_set:
            self.send_pi_msg(X, u_node)

    def send_pi_msg(self, Z, X):
        """
        Send a pi message from parent 'Z' to child 'X'
        Z : parent node
        X : child node
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
        Update the CPD for the given variable and value with the conditional variables and values

        Inputs:
            data_list:  [(variable, value, probability),(givenVar, givenVal),...,(givenVar, givenVal)]
        """
        # build var name index
        if not self.var_index:
            var_list = [data[0] for data in data_list]
            self.build_index(var_list)

        # get index of cpd to update
        prob_idx = [0]*len(data_list) # index of cpd to change
        for data in data_list:
            idx = self.var_index.index(data[0])
            prob_idx[idx] = data[1]
        alt_prob_idx = prob_idx[:]
        alt_prob_idx[0] = 1 - alt_prob_idx[0]  # flip the outcome
        prob_idx = tuple(prob_idx)
        alt_prob_idx = tuple(alt_prob_idx)

        # update cpd with probability value
        prob_value = data_list[0][2]
        self.cpd[prob_idx] = prob_value
        self.cpd[alt_prob_idx] = np.round(1 - prob_value, 6)


def read_problem_query(query_line):
    """
    Read line in project file to get problem number and query list
    """
    problem_query = {
        'question_number': query_line[0],
        'query_list': []
    }
    for qstr in query_line[1:]:
        problem_query['query_list'].append(qstr)  # add to query list
    return problem_query

def parse_query(qstr):
    """
    Parse a query string to extract the probability nodes and values
    and evidence nodes and values

    query string:
    qstr = 'P(A1, B0 | D0, F1)'
         = 'P(A1)'
         = 'P(A1|D1,F0)'
    """
    node_regex = '([a-z]+)\d' # with flags gi
    value_regex = '[a-z]+(\d)' # with flags gi
    X = [] # nodes to query
    E = [] # evidence nodes
    try:
        query = qstr.lstrip('P(').rstrip(')') # remove P( and ) from string
        query = query.split('|')  # split by query nodes and evidence nodes
        nodes = query[0].split(',')
        if len(query) > 1:
            evidence = query[1].split(',')
        else:
            evidence = []
        for node in nodes:
            n = re.search(node_regex, node, re.I).group(1)
            v = int(re.search(value_regex, node, re.I).group(1))
            X.append((n, v))
        for node in evidence:
            n = re.search(node_regex, node, re.I).group(1)
            v = int(re.search(value_regex, node, re.I).group(1))
            E.append((n, v))
    except:
        print('Invalid query string: "{}"'.format(qstr))
        raise
    return X, E


def read_file(fname):
    """
    Read problem file, generate graph and questions
    """
    G = None
    is_credal = False  # support for credal sets
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
                # print('line', line)
                data_list = []
                prob_str = line[-1]
                if ',' in prob_str:
                    # Draw from uniform distribution for intervals
                    is_credal = True
                    low, high = [float(x) for x in prob_str.split(',')]
                    prob = np.random.uniform(low, high)
                    # print('low={:.2f}, high={:.2f}, prob={:.4f}'.format(low,high,prob))
                else:
                    prob = float(prob_str)
                data_list.append((line[0][0], int(line[0][1]), prob))
                # print('data_list',data_list,'   line',line)
                for x in line[1:]:
                    if x == '|':
                        continue
                    if x == '=':
                        break
                    if x[0].isalpha() and x[1].isdigit():
                        data_list.append((x[0], int(x[1])))
                cpds.append(data_list)

            # Queries
            elif any([query_check.match(item) for item in line]) and (len(queries) < Q):
                prob_dict = read_problem_query(line)
                queries.append(prob_dict)

    # Create Graph object
    if (len(edges) == M)  and (len(cpds) == R):
        G = Graph(edges=edges)
        G.is_credal = is_credal  # set credal network flag
        for cpd in cpds:
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
        for query in queries:
            print('Problem {}'.format(query['question_number']))
            for q in query['query_list']:
                if not G.is_credal:
                    p = G.infer(q)
                    print('    {} = {:.4f}'.format(q, p))
                else:
                    # Re-create graph with drawn CPDs from credal sets
                    n = 10000   # number of samples
                    p = np.zeros(n)
                    for i in range(n):
                        p[i] = G.infer(q)
                        G, _ = read_file(fname)
                    print('    {} = [{:.4f}, {:.4f}]'.format(q, np.min(p), np.max(p)))
    else:
        print("Include filename 'project.txt' for Graph parameters as argument")
