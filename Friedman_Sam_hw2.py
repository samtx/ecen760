from __future__ import print_function
import sys

# Create a Graph object
class Graph(object):
    """ Graph object that has sets of nodes and edges """
    def __init__(self, edges=set(), nodes=set(), observations=set()):
        self.parents = {}
        self.children = {}
        self.nodes = nodes
        if len(nodes) > 0:
            self.size = len(nodes)
        else:
            self.size = 0
        self.edges = edges
        self.obs = observations

        self.build_parent_and_child_sets()
    
    def add_node(self, node):
        """ Add node to graph """
        if node not in self.nodes:
            self.size += 1
            self.nodes.add((node))
        else:
            pass
    
    def build_parent_and_child_sets(self):
        for edge in self.edges:
            u, v = edge
            self.add_node(u)
            self.add_node(v)
                
            if v not in self.parents:
                self.parents.update({v : set(u)})
            else:
                self.parents[v].add(u)

            if u not in self.children:
                self.children.update({u : set(v)})
            else:
                self.children[u].add(v)
                
    def get_parents(self, node):
        """ Return parents of node """
        if node in self.parents:
            return self.parents[node]
        else:
            return set()
    
    def get_children(self, node):
        """ Return children of node """
        if node in self.children:
            return self.children[node]
        else:
            return set()
                
    def find_ancestors(self, source_node):
        """ Use a breadth-first search to find all ancestors of node """
        visited = set()  # initialize ancestor set
        queue = [source_node]              
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend(self.get_parents(vertex)-visited)
                print(queue)
        return (visited - set(source_node))
    
    def find_descendants(self, source_node):
        """ Use a breadth-first search to find all descendants of node """
        visited = set()  # initialize descendant set
        queue = [source_node]              
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend(self.get_children(vertex)-visited)
                print(queue)
        return (visited - set(source_node))
        
                
    def d_separated(self, X, Z):
        """ 
        Find all nodes that are d_separated from node given observation
        Based on Algorithm 3.1, page 75, in PGM book
        
        X: starting node
        Z: set of observations
        
        """
        # Phase 1: insert all ancestors of Z into A
        L = Z.copy()  # set L to be the set of observations
        A = set()  # set A to be the empty set
#         print('%% A={}'.format(A),'  L={}'.format(L))
        while len(L) != 0:
            Y = L.pop()  # remove an item from the set
#             print('!! A={}'.format(A),'  Y={}'.format(Y),'  L={}'.format(L))
            if Y not in A:
                L = L.union(self.get_parents(Y))  # Y's parents need to be visited
            A = A.union(Y)  # Y is ancestor of evidence
        # print('$$ A={}'.format(A),'  Y={}'.format(Y),'  L={}'.format(L))

        # print('A={}'.format(A))
        # tmp = Z.copy()
        # for i in range(len(tmp)):
        #     tmp2 = tmp.pop()
        #     print('ances({})={}'.format(tmp2,self.find_ancestors(tmp2)))

        # Phase 2: traverse active trails starting from node
        L = {(X, 'up')}
        V = set()  # (node, direction) marked as visited
        R = set()  # Nodes reachable via active trail
        while len(L) != 0:
            # print('hello!!')
            # select some (Y, dr) from L
            (Y, dr) = L.pop()
            # print('(Y,dr)={}'.format((Y,dr)))
            # print('V={}'.format(V), '  Y={}'.format(Y), '  L={}'.format(L), '  Z={}'.format(Z))
            if (Y, dr) not in V:
                # print('&& R={}'.format(R),'  Y={}'.format(Y),'  Z={}'.format(Z),'  L={}'.format(L))
                if Y not in Z:
                    R = R | {Y}  # Y is reachable
                    # print('RRR={}'.format(R))
                V = V | {(Y,dr)}  # mark (Y, dr) as visited
                # print('V={}'.format(V),'  Y={}'.format(Y),'  L={}'.format(L),'  Z={}'.format(Z))
                # print('dr={}'.format(dr))
                if dr=='up' and Y not in Z:
                    # trail up through Y active if Y not in Z
                    # print('yoyoyo')
                    # print('parents of {} -> {}'.format(Y,self.get_parents(Y)))
                    tmp_iter= self.get_parents(Y)
                    # print('tmp_iter={}'.format(tmp_iter))
                    for z in tmp_iter:
                        # print('z={}'.format(z))
                        L = L.union({(z,'up')}) # Y's parents to be visited from bottom
                    # print('children of {} -> {}'.format(Y,self.get_children(Y)))
                    tmp_iter= self.get_children(Y)
                    # print('**tmp_iter={}'.format(tmp_iter))
                    for z in tmp_iter:
                        L = L.union({(z,'down')}) # Y's children to be visited from top
                        # print('z={}'.format(z),' L={}'.format(L))
                elif dr=='down':
                    # trails down through Y
                    if Y not in Z:
                        # downward trails to Y's children are active
                        for z in self.get_children(Y):
                            # print('z={}'.format(z))
                            L = L.union({(z,'down')})  # Y's children to be visited from top
                    if Y in A:
                        # v-structure trails are active
                        for z in self.get_parents(Y):
                            # print('z={}'.format(z))
                            L = L.union({(z,'up')}) # Y's parents to be visited from bottom
        return R


def read_file(fname):
    """ Read problem file, generate graph and questions """
    g_list = []  # list of graphs
    q_list = [] # list of problems
    edges = set()
    queries = []
    V, M, Q = 0, 0, 0
    with open(fname) as f:
        for raw_line in f:
            # split the line into components separated by whitespace
            line = raw_line.split()
            # skip '#' as comment
            if line[0][0] == '#':
                continue

            # New Graph description
            # NOTE: this doesn't work with Python 2.7
            elif all([x.isdigit() for x in line]):
                # print('beginning of graph description')
                # print(line)
                edges = set()
                queries = []
                V, M, Q = [int(x) for x in line]
                # V: number of nodes
                # M: number of edges
                # Q: number of queries

            # Edges
            elif all([x.isalpha() for x in line]) and len(edges) < M:
                u, v = line
                edges = edges | {(u, v)}

            # Queries
            elif line[1]=='|' and len(queries) < Q:
                y = line[0]   # source node Y
                z = {x for x in line[2:]}  # evidence set Z
                queries.append((y, z))

            # Create Graph object
            if (len(edges) == M) and (len(queries) == Q):
                print('Finished!!')
                G = Graph(edges=edges)

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

                g_list.append(G)
                q_list.append(queries)

    return g_list, q_list



if __name__ == "__main__":

    if len(sys.argv) > 1:
        # read filename as argument
        fname = sys.argv[1]
        g_list, q_list = read_file(fname)
        print(g_list[0].edges)
        print(q_list[0])

    else:
        # Build Graph from homework 1, problem 3
        edges = {('A', 'D'), ('B', 'D'), ('D', 'G'), ('D', 'H'), ('G', 'K'), ('H', 'K'),
                 ('H', 'E'), ('C', 'E'), ('E', 'I'), ('F', 'I'), ('F', 'J'), ('I', 'L'),
                 ('J', 'M')}
        G = Graph(edges=edges)
        Z = {'K', 'E'}
        dsep_nodes = G.d_separated('H', Z)
        print('dsep nodes:', dsep_nodes)

