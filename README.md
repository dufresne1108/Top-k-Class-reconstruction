# Top-k-Class-reconstruction for Incremental Learning
Official Implementation of "Top-k Class-reconstruction for Incremental Learning" 

This code provides an implementation for Top-k Class-reconstruction. This repository is implemented with pytorch and the scripts are written to run the experiments on GPUs.

## abstract
For a typical classification task, We usually get highest conﬁdence scores of the output layer of networks as prediction class. But for incremental learning, this can cause incremental over-confidence problem: the scores of new task higher than the correct one. In this paper, we present a top-k class-reconstruction framework that using class-decoder networks to reconstruct data from the output layer of representation learning networks. We select top-k candidate class-decoders by sorted confidence scores and use the confidence scores and reconstruction errors together for prediction. Our experiments demonstrate that the top-k class-reconstruction method can be used as a general method to improve the performance of the various base incremental-model.