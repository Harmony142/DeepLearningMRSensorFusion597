# PyTorch w/ CUDA on this device (GPU Power to run DNN)
# UMass Amherst ECE597SD: Deep-Learning Based Sensor Fusion for MR
# Group 3: Alexander Dickopf, Zachary Wannie, John Murray, Zhehang Zhang

# Based off tutorial for EKF implementation using Pyro and Torch libraries by Pyro API
# Reference: https://pyro.ai/examples/ekf.html

# Tested using ADVIO: An Authentic Dataset for Visual-Intertial Odometry: 
# Santiago Cortés, Arno Solin, Esa Rahtu, and Juho Kannala (2018). 
# ADVIO: An authentic dataset for visual-inertial odometry. 
# In European Conference on Computer Vision (ECCV). Munich, Germany.

# Import required libraries/classes

import csv
import os
import math

import torch
import pyro
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.contrib.tracking.extended_kalman_filter import EKFState
from pyro.contrib.tracking.distributions import EKFDistribution

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.7.0')
steps = 0

with open('accelerometer.csv') as file1:
    temp = csv.reader(file1, delimiter=',') 

    # Store XYZ of position here
    Px = []  
    Py = []  
    Pz = [] 

    # Gather data and store in Torch tensors
    for r in temp:
        Px.append(float(r[1]))  
        Py.append(float(r[2]))  
        Pz.append(float(r[3]))  
        
    Px = torch.tensor(Px)
    Py = torch.tensor(Py)
    Pz = torch.tensor(Pz)
    
    # Calculate mean and covariance for model training
    Px_var = torch.var(Px)
    Px_mean = torch.mean(Px)
    
    Py_var = torch.var(Py)
    Py_mean = torch.mean(Py)
    
    Pz_var = torch.var(Pz)
    Pz_mean = torch.mean(Pz)
    
    steps = r # Used for model construction
    
with open('gyro.csv') as file2:
    temp = csv.reader(file2, delimiter=',') 
    
    # Store XYZ of rotation here
    Rx = []  
    Ry = []  
    Rz = [] 

    # Gather data and store in Torch tensors
    for r in temp:
        Rx.append(float(r[1]))  
        Ry.append(float(r[2]))  
        Rz.append(float(r[3]))  
        
    Rx = torch.tensor(Rx)
    Ry = torch.tensor(Ry)
    Rz = torch.tensor(Rz)
    
    # Calculate mean and covariance for model traning
    Rx_var = torch.var(Rx)
    Rx_mean = torch.mean(Rx)
    
    Ry_var = torch.var(Ry)
    Ry_mean = torch.mean(Ry)
    
    Rz_var = torch.var(Rz)
    Rz_mean = torch.mean(Rz)
    
with open('magnetometer.csv') as file3:
    temp = csv.reader(file3, delimiter=',') 

    # Store XYZ of magnetometer here
    Mx = []  
    My = []  
    Mz = [] 

    # Gather data and store in Torch tensors 
    for r in temp:
        Mx.append(float(r[1]))  
        My.append(float(r[2]))  
        Mz.append(float(r[3]))  
        
    Mx = torch.tensor(Mx)
    My = torch.tensor(My)
    Mz = torch.tensor(Mz)
    
    # Calculate mean and covariance for model training
    Mx_var = torch.var(Mx)
    Mx_mean = torch.mean(Mx)
    
    My_var = torch.var(My)
    My_mean = torch.mean(My)
    
    Mz_var = torch.var(Mz)
    Mz_mean = torch.mean(Mz)

# 1. Manipulation of IMU data
# Format is .csv with 3-D coordinates of position, rotation, magnetometer
# e.g. Px, Py, Pz, Rx, Ry, Rz, Mx, My, Mz\n

# 2. Define our DNN model
# Model based around Extended Kalman Filter
# Will be using Pyro's predefined EKF classes to structure model
def model():

    # Calls below used per recommendation of Pyro
    # .clear_param_store helps to avoid leaking parameters from previous models
    pyro.set_rng_seed(0)
    pyro.clear_param_store

    # Model Structure
    # Sampling on seperate coordinates from 3 position, rotation, magnetometer
    
    # Observe Position
    pyro.sample('Xposition_{}'.format(i), EKFDistribution(Px_mean, Px_var, dynamic_model=None, 
    measurement_cov=0, time_steps=steps), obs=Px)
    pyro.sample('Yposition_{}'.format(i), EKFDistribution(Py_mean, Py_var, dynamic_model=None,
    measurement_cov=0, time_steps=steps), obs=Py)
    pyro.sample('Zposition_{}'.format(i), EKFDistribution(Pz_mean, Pz_var, dynamic_model=None,
    measurement_cov=0, time_steps=steps), obs=Pz)

    # Observe Rotation
    pyro.sample('XRotation_{}'.format(i), EKFDistribution(Rx_mean, Rx_var, dynamic_model=None,
    measurement_cov=0, time_steps=steps), obs=Rx)
    pyro.sample('YRotation_{}'.format(i), EKFDistribution(Ry_mean, Ry_var, dynamic_model=None,
    measurement_cov=0, time_steps=steps), obs=Ry)
    pyro.sample('ZRotation_{}'.format(i), EKFDistribution(Rz_mean, Rz_var, dynamic_model=None,
    measurement_cov=0, time_steps=steps), obs=Rz) 

    # Observe Magnetometer
    pyro.sample('XMagnetometer_{}'.format(i), EKFDistribution(Mx_mean, Mx_var, dynamic_model=None,
    measurement_cov=0, time_steps=steps), obs=Mx)
    pyro.sample('YMagnetometer_{}'.format(i), EKFDistribution(My_mean, My_var, dynamic_model=None,
    measurement_cov=0, time_steps=steps), obs=My)
    pyro.sample('ZMagnetometer_{}'.format(i), EKFDistribution(Mz_mean, Mz_var, dynamic_model=None,
    measurement_cov=0, time_steps=steps), obs=Mz)
    
# 3. Create MAP/MLE estimation guide
# Will either use MAP or MLE (currently MAP; MLE innacurrate for small data)
# Needed as parameter for SVI call
guide = AutoDelta(model)

# 4. Choose model optimizer
# Most common is Adam (learning rate set to default value; subject to change)
optim = pyro.optim.Adam({'lr': 1e-3})

# 5. Perform stochastic variable inference (SVI)
# Will use ELBO to compute loss for each step
svi = SVI(model, optim, loss=Trace_ELBO(retain_graph=True))
