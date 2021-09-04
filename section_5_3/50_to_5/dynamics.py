import numpy as np 

def angle_normalize(theta):
  x_plot = np.cos(theta)
  y_plot = np.sin(theta)
  angle = np.arctan2(y_plot,x_plot)
  return angle


def reward_func(x,u): #rewardがnumpy.array(1)型になっていると学習がうまくいかないことに注意．
    cost = -x[0,0]**(2) - 0.1*x[0,1]**(2) - 10.0*u**(2)
    return cost[0]

def Dynamics(x, u, a_param, b_param, l=1.0, g=9.81, delta=2**(-4)):
    theta = x[0,0]
    omega = x[0,1]

    reward = reward_func(x, u)
    done = False

    new_theta = theta + delta*(omega)
    new_omega = omega + delta*(g/l*np.sin(theta) - a_param*omega + b_param*u[0])

    new_theta = angle_normalize(new_theta)
    new_x = np.array([[new_theta, new_omega]])

    return new_x, reward, done

def Initialize():
    theta = np.pi # Get the initial state s_0
    omega = 0.0 
    state = np.array([[theta, omega]]) #numpy array (1,2)

    return state
