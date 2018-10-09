Readme for Sam Friedman HW 2 submission
10/9/2018

This program has been tested on Python 3.5.2 and 2.7.12.

There are no additional non-standard libraries required to run the program.

It accepts a text file as the first argument and will print out the results to stdout as described by the TA.
All the results from the program match the solutions to homework 3.

Example:

$ python Friedman_Sam.py test.txt

Output from program:
#######

sam@sam-Inspiron-3543:~/code/ecen760(master)$ python Friedman_Sam_hw2.py
Check answers from HW 1, Problem 3:
(3a) Active trail from A to J given set(['L', 'G'])? True
(3b) Active trail from A to C given set(['L'])? True
(3c) Active trail from G to L given set(['D'])? False
(3d) Active trail from G to L given set(['K', 'M', 'D'])? True
(3e) Active trail from B to F given set(['C', 'L', 'G'])? True
(3f) Nodes d-separated from A given set(['K', 'E']) = set(['I', 'J', 'M', 'L', 'F'])
(3g) Nodes d-separated from B given set(['L']) = set([])
sam@sam-Inspiron-3543:~/code/ecen760(master)$ python Friedman_Sam_hw2.py test.txt
None
None
A C B E F I H J M L
A C B J F
None
sam@sam-Inspiron-3543:~/code/ecen760(master)$
