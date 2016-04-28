# nnexp2
neural network experiment 2, deep q (like) learning

Written from scratch, this demo is an implementation of a neural network is trained to navigate an area towards a fixed goal. Here we use a deep q learning (like) algorithm variant based on the paper http://arxiv.org/pdf/1312.5602v1.pdf.

Important things to note is that q learning tranis the neural net a little diferently than you might expect. The implementation uses 'temporal windows', which are essentially a sequence of saved states. The temporal windows are saved in a list, and then are fed back into the input nodes of the network and used to train the network against the expected output. Temporal window sequences are chosen randomly to aid learning distribution (see the paper for more detailed explanation as to why).

To run you need to have nodejs installed (v4.2.4 +). From the command line type node index.js
