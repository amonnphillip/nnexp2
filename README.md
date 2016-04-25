# nnexp2
neural network experiment 2, deep learning

Written from scratch, this demo is an implementation of a neural network is trained to navigate an area towards a fixed goal. Here we use a deep q learning algorithm variant based on the paper http://arxiv.org/pdf/1312.5602v1.pdf.

Important things to note is the implementation uses 'temporal windows', which are essentially a sequence of saved states. These states are fed back into the input nodes of the network and used to train the network for the task at hand. Which states are used to train are chosen randomly (see the paper to know why).

To run you need to have nodejs installed (v4.2.4 +). From the command line type node index.js
