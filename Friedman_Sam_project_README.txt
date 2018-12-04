Readme for Sam Friedman ECEN 760 Project submission
12/2/2018

This program has been tested on Python 3.5.2 and Numpy 1.15.4.

The additional non-standard Python packages are required:
    - numpy >= 1.15.4

Files in the directory:
(1) Friedman_Sam_project.py         --> main Python program for Graph analysis and Pearl's MP algorithm implementation
(2) project.txt                     --> text file containing graph edges and CPD parameters and problem inference queries
(3) test_project.py                 --> test suite file to confirm program works as desired
(4) Friedman_Sam_project_README.txt --> this README file

Run the test suite to confirm that the program works as desired

$ python test_project.py -v

Output:
######################
test_01 (__main__.TestPearlAlgorithm) ... ok
test_02 (__main__.TestPearlAlgorithm) ... ok
test_03 (__main__.TestPearlAlgorithm) ... ok
test_04 (__main__.TestPearlAlgorithm) ... ok
test_05 (__main__.TestPearlAlgorithm) ... ok
test_06 (__main__.TestPearlAlgorithm) ... ok
test_07 (__main__.TestPearlAlgorithm) ... ok
test_08 (__main__.TestPearlAlgorithm) ... ok
test_09 (__main__.TestPearlAlgorithm) ... ok
test_10 (__main__.TestPearlAlgorithm) ... ok
test_11 (__main__.TestPearlAlgorithm) ... ok
test_12 (__main__.TestPearlAlgorithm) ... ok
test_13 (__main__.TestPearlAlgorithm) ... ok
test_14 (__main__.TestPearlAlgorithm) ... ok
test_15 (__main__.TestPearlAlgorithm) ... ok

----------------------------------------------------------------------
Ran 15 tests in 0.026s

OK

#######################

The main program (Friedman_Sam_project.py) accepts a text file as the first argument and will print out the results to stdout.
The output is the answer to the project problem (1b)

Example:
$ python Friedman_Sam_project.py project.txt

Output from program:
#######
Problem (i)
    P(A1|B0) = 0.7000
    P(A1|D0) = 0.7687
    P(A1|D0,B0) = 0.7656
    P(A1|D0,G1) = 0.7687
Problem (ii)
    P(B1|A1) = 0.4000
    P(B1|C1) = 0.6842
    P(B1|A1,C1) = 0.6667
    P(B1|C1,F0) = 0.6842
Problem (iii)
    P(C1) = 0.4560
    P(C1|A1) = 0.5400
    P(C1|A1,B0) = 0.3000
    P(C1|D0) = 0.7458
    P(C1|D0,F0) = 0.7458
Problem (iv)
    P(D1) = 0.5720
    P(D1|E0) = 0.6523
    P(D1|C0,E0) = 0.8000
    P(D1|B1,G0) = 0.8066
    P(D1|B1,G0,F1) = 0.9669
Problem (v)
    P(E1) = 0.3824
    P(E1|C1) = 0.6000
    P(E1|F0) = 0.4359
    P(E1|C1,F0) = 0.6000
    P(E1|A1,B1) = 0.5600
Problem (vi)
    P(F1) = 0.4432
    P(F1|A1) = 0.4180
    P(F1|A1,C0) = 0.5800
    P(F1|A1,C0,E0) = 0.5800
    P(F1|B1,G0) = 0.5839
Problem (vii)
    P(G1) = 0.6140
    P(G1|C0) = 0.5000
    P(G1|C0,D0) = 0.9000
    P(G1|E0) = 0.5738
    P(G1|A0,B1) = 0.6250
Problem (viii)
    P(A1,D1|F0,B1) = 0.1124
    P(C0,E1|F1,G0) = 0.1504
    P(F0,B1|G0,E1) = 0.2016
    P(G1,B0|F1,A0) = 0.2779