import unittest
from Friedman_Sam_project import read_file

# import and generate graph
G, _ = read_file('project.txt')

class TestPearlAlgorithm(unittest.TestCase):

    def test_01(self):
        q = (([('A', 1)], [('B', 0)]), 0.7)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_02(self):
        q =  (([('A', 1)], [('D', 0)]), 0.7687)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_03(self):
        q = (([('B', 1)], [('A', 1)]), 0.4)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_04(self):
        q = (([('B', 1)], [('C', 1)]), 0.6842)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_05(self):
        q = (([('C', 1)], []), 0.456)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_06(self):
        q = (([('C', 1)], [('A', 1)]), 0.54)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_07(self):
        q = (([('D', 1)], []), 0.572)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_08(self):
        q = (([('D', 1)], [('E', 0)]), 0.6523)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_09(self):
        q = (([('E', 1)], []), 0.3824)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_10(self):
        q = (([('E', 1)], [('C', 1)]), 0.6)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_11(self):
        q = (([('F', 1)], []), 0.4432)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_12(self):
        q = (([('F', 1)], [('A', 1)]), 0.4180)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_13(self):
        q = (([('G', 1)], []), 0.6140)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_14(self):
        q = (([('G', 1)], [('C', 0)]), 0.5)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)

    def test_15(self):
        q = (([('A', 1), ('D', 1)], [('F',0), ('B', 1)]), 0.1124)
        X, E = q[0]
        self.assertAlmostEqual(G.infer(X, E), q[1], places=4)






if __name__ == "__main__":
    unittest.main()