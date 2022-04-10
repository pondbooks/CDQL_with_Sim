### Source Code of Continuous Deep Q-Learning with Simulator for Stabilization of Uncertain Discrete-Time Systems

Pre-train DNN's paramete vectors are in weight_DNN.
https://arxiv.org/abs/2101.05640

#### warning (2022/4/10)
In this source code, we use `A**(-1)` for computing an inverse matrix. However, in this case, we must define the matrix as `np.matrix`. So, we should change `A**(-1)` to `np.linalg.inv()`. In this example, fortunately, we consider the 1-dim problem. We can obtain the same result. 

![warning](https://user-images.githubusercontent.com/68591842/162602207-06bd45c7-ea50-49e6-9307-384ac013422a.png)
