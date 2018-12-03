# Sam Friedman
# ECEN 760
# Project
# 12/2/2018

# Test Suite

import unittest   # standard library
from Friedman_Sam_project import read_file

# import and generate graph
G, _ = read_file('project.txt')

class TestPearlAlgorithm(unittest.TestCase):

    def test_01(self):
        qstr, p = 'P(A1|B0)', 0.7
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_02(self):
        qstr, p = 'P(A1|D0)', 0.7687
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_03(self):
        qstr, p = 'P(B1|A1)', 0.4
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_04(self):
        qstr, p = 'P(B1|C1)', 0.6842
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_05(self):
        qstr, p = 'P(C1)', 0.456
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_06(self):
        qstr, p = 'P(C1|A1)', 0.54
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_07(self):
        qstr, p = 'P(D1)', 0.572
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_08(self):
        qstr, p = 'P(D1|E0)', 0.6523
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_09(self):
        qstr, p = 'P(E1)', 0.3824
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_10(self):
        qstr, p = 'P(E1|C1)', 0.6
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_11(self):
        qstr, p = 'P(F1)', 0.4432
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_12(self):
        qstr, p = 'P(F1|A1)', 0.4180
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_13(self):
        qstr, p = 'P(G1)', 0.6140
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_14(self):
        qstr, p = 'P(G1|C0)', 0.5
        self.assertAlmostEqual(G.infer(qstr), p, places=4)

    def test_15(self):
        qstr, p = 'P(A1,D1|F0,B1)', 0.1124
        self.assertAlmostEqual(G.infer(qstr), p, places=4)


if __name__ == "__main__":
    unittest.main()