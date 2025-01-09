import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import time

d = 2

# F(x) in the nonlinear dynamics
def drift(x):
    return np.zeros((N*2,1))

# G(x) in the nonlinear dynamics
def control(x):
    return np.eye(N*2)

# Collection of CBFs
def h(x):
    out = []
    for i in range(N):
        for j in range(i+1,N):
            xi = x[i*d:(i+1)*d]
            xj = x[j*d:(j+1)*d]
            hij = np.linalg.norm(xi - xj)**2 - safe_dist**2
            out.append(hij)
    return out

# Collection of their gradients
def gradh(x):
    out = []
    for i in range(N):
        for j in range(i+1,N):
            xi = x[i*d:(i+1)*d]
            xj = x[j*d:(j+1)*d]
            eij = np.zeros((N*d,1))
            eij[i*d:(i+1)*d] = 2*(xi - xj)
            eij[j*d:(j+1)*d] = 2*(xj - xi)
            out.append(eij)
    return out

# Collection of their Hessians
def hessh(x):
    out = []
    eyed = np.eye(d)
    for i in range(N):
        for j in range(i+1,N):
            eij = np.zeros((N*d,N*d))
            eij[i*d:(i+1)*d,i*d:(i+1)*d] = 2*eyed
            eij[j*d:(j+1)*d,j*d:(j+1)*d] = 2*eyed
            eij[i*d:(i+1)*d,j*d:(j+1)*d] = -2*eyed
            eij[j*d:(j+1)*d,i*d:(i+1)*d] = -2*eyed
            out.append(eij)
    return out

# Smooth saturation function
def sat(x,thresh):
    return thresh*np.tanh(x/thresh)

# Jacobian of the smooth saturation function
def jacsat(x, thresh):
    return np.diagflat(1 - np.tanh(x/thresh)**2)

# Nominal feedback controller
def unom(x,K):
    return -np.kron(np.eye(N),K)@(x - xd)

# Jacobian of the Nominal feedback controller
def jacunom(x,K):
    return -np.kron(np.eye(N),K)

def eta(t):
    return np.exp(-0.3*t)

def etadot(t):
    return -0.3*eta(t)

# Gradient of the objective function
def gradf(u,x,t):
    # this is correct in the single integrator case. Change it if we change dynamics
    grads = gradh(x)
    hs = h(x)
    out = u - unom(x,K)
    for i in range(len(hs)):
        out -= eta(t)*grads[i] / (cbf_scale*hs[i] + np.dot(grads[i].T, u))
    return out

# Hessian of the objective function
def hessf(u,x,t):
    # this is correct in the single integrator case. Change it if we change dynamics
    grads = gradh(x)
    hs = h(x)
    out = np.eye(N*d)
    for i in range(len(hs)):
        out += eta(t) * grads[i] @ grads[i].T / ((cbf_scale*hs[i] + np.dot(grads[i].T,u))**2)
    return out

# Partial^2 (Objective) / (Partial u, Partial x)
def partialf_ux(u,x,t):
    # this is correct in the single integrator case. Change it if we change dynamics
    Kout = np.kron(np.eye(N), K)
    hs = h(x)
    grads = gradh(x)
    hesshs = hessh(x)
    out = Kout
    for i in range(len(hs)):
        out += eta(t) * ((cbf_scale*hs[i] + np.dot(grads[i].T, u))*hesshs[i] - grads[i] @ (grads[i].T + u.T @ hesshs[i]))/((cbf_scale*hs[i] + np.dot(grads[i].T, u))**2)
    return out
    
# Partial^2 (Objective) / (Partial u, Partial eta)
def partialf_ueta(u,x,t):
    # this is correct in the single integrator case. Change it if we change dynamics
    hs = h(x)
    grads = gradh(x)
    out = np.zeros_like(grads[0])
    for i in range(len(hs)):
        out += grads[i] / (cbf_scale*hs[i] + np.dot(grads[i].T, u))
    return out

# Derivatives of the time-varying parameters x, eta
def thetadot(u,x,t):
    # this is only correct in the single integrator case
    return np.vstack((u,etadot(t)))

# Jacobian of the dynamics with respect to the state u
def jac_Fu(u,x,t):
    y = u - gamma*gradf(u,x,t)
    return -np.eye(N*d) + jacsat(y,safe_speed) * (np.eye(N*d) - gamma*hessf(u,x,t))

# Jacobian of the dynamics with respect to the time-varying parameters
def jac_Ftheta(u,x,t):
    y = u - gamma*gradf(u,x,t)
    return -jacsat(y,safe_speed) @ np.hstack((gamma*partialf_ux(u,x,t), gamma*partialf_ueta(u,x,t)))

# Dynamics for the coupled system
def ode_system(u, x, t):
    y = u - gamma*gradf(u,x,t)
    feedforward = -np.linalg.solve(jac_Fu(u,x,t), jac_Ftheta(u,x,t) @ thetadot(u,x,t))
    udot = (-u + sat(y,safe_speed)) + feedforward
    xdot = u
    return udot, xdot

# Instantiate Robotarium object

N = 4
gamma = 0.5
safe_dist = 0.5
safe_speed = 0.12
cbf_scale = 3

K = np.array([[1, 0.], [0., 1]])

initial_conditions = np.array([[-2.5, -2.5, 2.5, 2.5], [-2.0, 2.0, 2.0, -2.0], [-3*np.pi/4, 3*np.pi/4, np.pi/4, -np.pi/4]])
normalizing_const = 0.8/2.2 # ensures that we are inside the Robotarium
initial_conditions[0:-1,:] *= normalizing_const
initial_conditions[-1,:] += np.pi + 0.01

xd = np.array([[2.0],[2.02],[2.0],[-2.05],[-2],[-2.05],[-2.0],[2.05]])

xd *= normalizing_const
goal_points = np.reshape(xd, (2,N), order='F')


safe_dist *= normalizing_const
print("safe distance: ", safe_dist)

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)

# initial control
u = np.zeros((2*N,1))

# Plotting Parameters
CM = np.array([[0,0,0], [255, 153, 51], [255, 51, 255], [51, 51, 255.]]) / 255
safety_radius_marker_size = determine_marker_size(r,0.1) # Will scale the plotted markers to be the diameter of provided argument (in meters)
font_height_meters = 0.2
font_height_points = determine_font_size(r,font_height_meters) # Will scale the plotted font height to that of the provided argument (in meters)

x_unis = np.reshape(initial_conditions[0:-1,:], (2*N,1), order = 'F')

# Create unicycle controller (not used)
unicycle_pose_controller = create_hybrid_unicycle_pose_controller()

# Create single integrator position controller
single_integrator_position_controller = create_si_position_controller()

# Create barrier certificates to avoid collision
#si_barrier_cert = create_single_integrator_barrier_certificate()
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
uni_barrier_cert = create_unicycle_barrier_certificate()

_, uni_to_si_states = create_si_to_uni_mapping()

# Create mapping from single integrator velocity commands to unicycle velocity commands
si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

# define x initially
x = r.get_poses()
x_si = uni_to_si_states(x)
r.step()

# Define number of iterations we will run
num_iterations = 1500
xs = np.zeros((2*N, num_iterations))
us = np.zeros((2*N, num_iterations))
hs = np.zeros((N*(N-1)//2, num_iterations))
ts = np.zeros(num_iterations)

# Define initial time
t_offset = time.time()
t_current = 0

for t in range(num_iterations):
    # Get poses of agents
    x = r.get_poses()

    # Get the single integrator state    
    x_si = uni_to_si_states(x)


    x_si_stacked = np.reshape(x_si, (2*N,1), order='F')

    xs[:,t] = np.squeeze(x_si_stacked)
    us[:,t] = np.squeeze(u)
    hs[:,t] = h(x_si_stacked)
    ts[t] = t_current
    
    x_unis = np.hstack((x_unis, np.reshape(x[0:-1,:], (2*N,1), order='F')))
    
    # Compute udot and xdot
    du, dx = ode_system(u, x_si_stacked, t_current)

    dxi = np.reshape(dx, (2, N), order='F')

    # Use robotarium to ensure robots don't leave the boundaries of the domain    
    dxi = si_barrier_cert(dxi, x_si)

    # Transform single integrator velocity commands to unicycle
    dxu = si_to_uni_dyn(dxi, x)
    # Send control inputs
    r.set_velocities(np.arange(N), dxu)
    
    # Iterate the simulation
    dt = time.time() - t_offset - t_current
    u += dt*du
    t_current += dt
    if np.linalg.norm(x_si-goal_points) < 0.05:
        r.step()
        break
    r.step()


xs = xs[:,0:t+1]
us = us[:,0:t+1]
hs = hs[:,0:t+1]
ts = ts[0:t+1]

x_unis = x_unis[:,0:t+1]

# Save states, controls, barrier function values, and times
np.save('si_states', xs)
np.save('controls', us)
np.save('h_values', hs)
np.save('times', ts)

print('minimum value of h:', np.min(hs))

x = r.get_poses()

# Plot for visualization at the end
starts = np.reshape(x_unis[:,0], (2,N), order='F')

r.axes.scatter(starts[0,:], starts[1,:], s=np.pi/4*safety_radius_marker_size, marker='o', facecolors='none',edgecolors=CM,linewidth=7)
r.axes.scatter(goal_points[0,:], goal_points[1,:], s=np.pi/4*safety_radius_marker_size, marker='*', facecolors='none',edgecolors=CM,linewidth=7)
for i in range(N):
    r.axes.plot(x_unis[i*d, :], x_unis[i*d+1, :], color=CM[i,:], linewidth=5)

r.step()
time.sleep(3)

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()