### Source Code of Continuous Deep Q-Learning with Simulator for Stabilization of Uncertain Discrete-Time Systems

Pre-train DNN's paramete vectors are in weight_DNN.

https://arxiv.org/abs/2101.05640

The example of a discrete-time system is a pendulum dynamics computed by **Euler-method** with stepsize ``2**(-4)``. We describe the dynamics in ``Pendulum.pdf``. The range of an angle is ``[-np.pi, np.pi]``.

![スクリーンショット 2022-04-28 8 54 00](https://user-images.githubusercontent.com/68591842/165649603-357b407c-b8fb-46be-bee6-0e1ed5f9dd33.png)

#### Normalize an angle

We use the ``np.arctan2(y, x)`` function (https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html) for normalization of an angle parameter. The range of the angle parameter is ``[-np.pi, np.pi]``. At first, the ``angle_normalize`` obtains an angle parameter theta. Secondly, the function computes the y-coordinate ``y`` and x-coordinate ``x`` by ``np.sin(theta)`` and ``np.cos(theta)``, respectively. Finally, the function computes the normalized angle by ``np.arctan2(y,x)``.

```
def angle_normalize(theta):
  x_plot = np.cos(theta)
  y_plot = np.sin(theta)
  angle = np.arctan2(y_plot,x_plot)
  return angle
```

![image](https://user-images.githubusercontent.com/68591842/165648899-3a8b65e0-ff0c-466b-bf4d-51c83e4e3e16.png)

This Fig. shows the angle normalize function in the range ``[-5*np.pi, 5*np.pi]``. The red lines show the angles = ``-5*np.pi``, ``-3*np.pi``, ``-np.pi``, ``np.pi``, ``3*np.pi``, ``5*np.pi``. The blue lines show the normalized angles = ``-3*np.pi``, ``3*np.pi``.

#### warning (2022/4/10)
In this source code, we use `A**(-1)` for computing an inverse matrix. However, in this case, we must define the matrix as `np.matrix`. So, we should change `A**(-1)` to `np.linalg.inv()`. In this example, fortunately, we consider the 1-dim problem. We can obtain the same result. 

![warning](https://user-images.githubusercontent.com/68591842/162602207-06bd45c7-ea50-49e6-9307-384ac013422a.png)
