import numpy as np
from continuous_cartpole import angle_normalize  # (((x+np.pi) % (2*np.pi)) - np.pi)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math



# def reward (env):
#     x, x_dot, theta, theta_dot = env.state
#     true_theta = angle_normalize(theta)
#
#     if math.fabs(x) > 2.3:
#         ###### this is the best
#         #r_x = -100
#         r_x = -1000
#     else:
#         r_x = 0
#     r_bonus = 0
#     if math.fabs(true_theta) < 0.3:
#         r_bonus = 5* np.cos(true_theta) - 0.5 *  theta_dot**2
#     #return r_bonus + r_x + 2*np.cos(true_theta) - 0.5* math.fabs(np.sin(true_theta)) - 0.001 * x_dot**2 - 0.001 * theta_dot**2 - 0.001 * x**2
#     #return r_x + (x_dot**2)+0.001*(x**2)+0.1*(true_theta**2) + r_bonus
#
#     ########## this is the best
#     return 2* np.cos(theta) - 0.01* math.fabs(np.sin(true_theta)*x_dot**2) - 0.001 * theta_dot** 2 - 0.0001 * x_dot**2  + r_x + r_bonus



# def reward(env):
#     x, x_dot, theta, theta_dot = env.state
#     true_theta = angle_normalize(theta)
#
#     if math.fabs(x) > 2.35:
#         r_x = -1000
#     else:
#         r_x = 1
#     r_bonus = 0
#     if math.fabs(true_theta) < 0.3:
#         r_bonus = 5* np.cos(true_theta) - 0.5 *  theta_dot**2
#     r_x_dot = - 0.001 * math.fabs(x_dot)
#     r_theta = np.cos(true_theta) - abs(np.sin(theta) * x_dot + 0.01 * x**2)
#     r_thetadot = -0.001 * theta_dot**2
#
#     return r_x + r_theta + r_x_dot + r_thetadot + r_bonus


# def reward(env):
#     x, x_dot, theta, theta_dot = env.state
#     true_theta = angle_normalize(theta)
#
#     if math.fabs(x) > 2.35:
#         r_x = -1000
#     else:
#         r_x =0
#     if math.fabs(true_theta) < 0.3:
#         r = 3* np.cos(true_theta) - 0.5 *  theta_dot**2 + r_x
#     else:
#         r =  np.cos(true_theta) - 0.01 * theta_dot * 2 - 0.001 * x_dot** 2 + r_x
#
#     return r

# def reward(env):
#     x, x_dot, theta, theta_dot = env.state
#     true_theta = angle_normalize(theta)
#
#     r_x, r_theta, r_thetadot, r_x_dot = 0, 0, 0, 0
#     if math.fabs(x) > 2.35:
#         r_x = -1000
#     else:
#         r_x = 1
#
#     if np.fabs(true_theta) > (np.pi / 2):
#         r_x_dot = - 0.001 * np.fabs(x_dot)
#         r_theta = np.cos(true_theta) - abs(np.sin(theta) * x_dot * x**2)
#         r_thetadot = -0.001 * theta_dot ** 2
#     elif np.fabs(true_theta) < 0.3:
#         r_theta = np.cos(true_theta) - 0.03 * theta_dot**2
#     else:
#         r_x_dot = - 0.01 * x_dot**2
#         r_theta = np.cos(true_theta)
#         r_thetadot = -0.01 * theta_dot ** 2
#     return r_x + r_theta + r_thetadot + r_x_dot

def reward_old(env):
    x, x_dot, theta, theta_dot = env.state
    true_theta = angle_normalize(theta)

    r_x, r_theta, r_thetadot, r_x_dot = 0, 0, 0, 0
    if math.fabs(x) > 2.35:
        r_x = -1000
    else:
        r_x = 1

    if np.fabs(true_theta) > np.pi / 2:
        # Under
        r_x_dot = - 0.001 * np.fabs(x_dot)
        r_theta = np.cos(true_theta) - abs(np.sin(theta) * x_dot * x**2)
        r_thetadot = -0.001 * theta_dot ** 2
    elif np.fabs(true_theta) < 0.3:
        # Close
        r_theta = gaussian(theta, 0, 0.5) + gaussian(theta_dot, 0, 0.5) + 1
    else:
        # Up not so close
        r_x_dot = - 0.01 * x_dot**2
        r_theta = np.cos(true_theta)
        r_thetadot = -0.01 * theta_dot ** 2
    return r_x + r_theta + r_thetadot + r_x_dot

def reward_new(env):
    x, x_dot, theta, theta_dot = env.state
    true_theta = angle_normalize(theta)

    r_x, r_theta, r_thetadot, r_x_dot = 0, 0, 0, 0
    if math.fabs(x) > 2.35:
        r_x = -1000
    else:
        r_x = 1

    if np.fabs(true_theta) > np.pi / 2:
        # Under
        #r_x_dot = - 0.001 * np.fabs(x_dot)
        r_theta = -1
        r_thetadot = -0.01 * theta_dot ** 2
    elif np.fabs(true_theta) < 0.3:
        # Close
        r_theta = 100 + -0.01 * theta_dot ** 2
    else:
        # Up not so close
        r_x_dot = - 0.01 * x_dot**2
        r_theta = np.cos(true_theta)
        r_thetadot = -0.01 * theta_dot ** 2
    return r_x + r_theta + r_thetadot + r_x_dot

def reward(env):
    x, x_dot, theta, theta_dot = env.state
    theta_norm = angle_normalize(theta)

    r_x, r_theta, r_thetadot, r_x_dot = 0, 0, 0, 0
    if math.fabs(x) > 2.4:
        return -100
    elif(abs(theta_norm)<np.pi/8):
        return  2*np.cos(theta_norm) - theta_dot**2 + 4
    
    elif(abs(theta_norm)<np.pi/2):
        return  2*np.cos(theta_norm) - 0.1*theta_dot**2 + 2
    else:
        return  np.cos(theta_norm) - 0.1*theta_dot**2*np.cos(theta_norm) - 0.01*x**2

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(-10, 10, 100)  # For X
    Y = np.linspace(-10, 10, 100)  # For X_dot
    X, Y = np.meshgrid(X, Y)
    zs = np.array([reward_new((np.squeeze(np.ravel(X[i, j])), np.squeeze(np.ravel(Y[i, j])), 0, 0))
                   for i in range(X.shape[0]) for j in range(X.shape[1])])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X')
    ax.set_ylabel('X_dot')
    ax.set_zlabel('Reward')
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(- np.pi, np.pi, 100)  # For theta
    Y = np.linspace(-10, 10, 100)  # For theta_dot
    X, Y = np.meshgrid(X, Y)
    zs = np.array([reward_new((0, 0, np.squeeze(np.ravel(X[i, j])), np.squeeze(np.ravel(Y[i, j]))))
                   for i in range(X.shape[0]) for j in range(X.shape[1])])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('Theata')
    ax.set_ylabel('Theta_dot')
    ax.set_zlabel('Reward')
    plt.show()
