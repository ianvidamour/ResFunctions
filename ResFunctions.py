#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 10:37:02 2023

@author: Ian Vidamour

A package of dynamic systems models for signal transformation, as well as some 
neural network packages for building simple supervised learning paradigms.
I've tried to annotate fully, but in case of any difficulties/confusions with 
the code please let me know at i.vidamour@sheffield.ac.uk

"""

import torch 
import torch.nn as nn
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numba
from Sparce import *

if torch.cuda.is_available() == True:
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)

'''

Standard ESN with leaky integrator neurons

'''

# Return weight matrices for an ESN
def InitialiseESN(Nin, Nnodes, sparsity):
    with torch.no_grad():
        # Randomly sample input/reservoir weights
        Win = torch.rand((Nin, Nnodes))
        Wres = torch.rand((Nnodes, Nnodes))
        # Set a percentage of reservoir weights to zero
        Wres[Wres<sparsity] = 0
        # Caclulate eigenvalues of reservoir weights and normalise by largest
        vals, vecs = np.linalg.eig(Wres.cpu())
        Wres = Wres/np.abs(vals[0])
    return Win, Wres
    
# Run a simple ESN with leaky integrator neurons
# s = input signal, Win = input weights, Wres =  reservoir weights,
# alpha = leak rate, rho = spectral radius, gamma = input scaling
def RunESN(s, Win, Wres, alpha, rho, gamma):
    with torch.no_grad():
        # Create output matrix
        x = torch.zeros((Wres.size(0), len(s)))
        # Perform iterative reservoir update equation
        for t in range(len(s)-1):
            x[:, t+1] = (1-alpha) * x[:, t] + alpha * torch.tanh(rho * torch.matmul(x[:, t], Wres) + gamma * torch.matmul(s[t, :], Win))
    return x

'''

Driven damped harmonic oscillator functions

'''

# Initialisation function for DDHO
def Init_DDHO(F, k, c, m, samplerate):
    # Calculate undamped resonant frequency, damping constant, damped res freq
    omega0 = np.sqrt(k/m)
    zeta = c/(2*np.sqrt(m*k))
    omegaz = omega0 * np.sqrt(1-zeta**2)
    # Create timebase
    t = np.linspace(0, omegaz*len(F)/samplerate, len(F))
    # Set initial conditions
    X = [0, 1, F[0], m, zeta, omega0]
    return X, t, omega0
    
# Define ODE for DDHO
def dXdt_DDHO(X, t):
    [x, v, F, m, zeta, omega0] = X
    return v, F/m - 2*zeta*omega0*v - x*omega0**2, F, m, zeta, omega0

# Initialise and run a driven damped harmonic oscillator, where 
# s = input signal (should be pre-multiplexed), k = stiffness, c = damping,
# m = mass, gamma = input scaling.
# By default, system is driven at damped resonant frequency.
def RunDDHO(s, k, c, m, gamma, samplerate):
    # Convert force into sinusoidal oscillation
    Fsig = np.zeros(len(s)*samplerate)*gamma
    cosot = np.cos(np.linspace(0, 2*np.pi, samplerate))
    for i in range(len(s)):
        Fsig[i*samplerate:(i+1)*samplerate] = cosot*s[i]
    # Initialise conditions
    X, t, omega0 = Init_DDHO(Fsig, k, c, m, samplerate)
    Xout = []
    Vout = []
    for dt in range(len(t)-1):
        if dt == 0:
            # Solve with initial conditions and update input for next timestep
            solver = odeint(dXdt_DDHO, y0=X, t=t[0:2])
            Xout.append(solver[1][0])
            Vout.append(solver[1][1])
            X[0] = solver[1][0]
            X[1] = solver[1][1]
            X[2] = Fsig[dt+1]
        else:
            # 
            solver = odeint(dXdt_DDHO, y0=X, t=t[dt-1:dt+1])
            Xout.append(solver[1][0])
            Vout.append(solver[1][1])
            X[0] = solver[1][0]
            X[1] = solver[1][1]
            X[2] = Fsig[dt+1]
    Xout.append(solver[1][0])
    Xout.append(solver[1][1])
    return Xout, Vout

# As above, but with a delayed feedback term which adds a percentage of past
# output on top of current input. Feedback strength controlled by fs.
def RunDDHO_fb(s, k, c, m, gamma, fs, samplerate):
    # Convert force into sinusoidal oscillation
    Fsig = np.zeros(len(s)*samplerate)*gamma
    cosot = np.cos(np.linspace(0, 2*np.pi, samplerate))
    for i in range(len(s)):
        Fsig[i*samplerate:(i+1)*samplerate] = cosot*s[i]
    # Initialise conditions
    X, t, omega0 = Init_DDHO(Fsig, k, c, m, samplerate)
    Xout = []
    Vout = []
    for dt in range(len(t)-1):
        if dt == 0:
            # Solve with initial conditions and update input for next timestep
            solver = odeint(dXdt_DDHO, y0=X, t=t[0:2])
            Xout.append(solver[1][0])
            Vout.append(solver[1][1])
            X[0] = solver[1][0]
            X[1] = solver[1][1]
            X[2] = Fsig[dt+1]
        elif dt < samplerate:
            # Solve as standard if input has not completed full cycle
            solver = odeint(dXdt_DDHO, y0=X, t=t[dt-1:dt+1])
            Xout.append(solver[1][0])
            Vout.append(solver[1][1])
            X[0] = solver[1][0]
            X[1] = solver[1][1]
            X[2] = Fsig[dt+1]
        else:
            # Solve with added feedback
            solver = odeint(dXdt_DDHO, y0=X, t=t[dt-1:dt+1])
            Xout.append(solver[1][0])
            Vout.append(solver[1][1])
            X[0] = solver[1][0]
            X[1] = solver[1][1]
            X[2] = Fsig[dt+1]+fs*Xout[dt-samplerate+1]
    Xout.append(solver[1][0])
    Vout.append(solver[1][1])
    return Xout, Vout


'''

Mackey-Glass oscillator functions

'''
         
# Dynamics of Mackey-Glass function. x = current input, xdelay = delayed input
# gamma = input scaling, eta = feedback strength, p = 'nonlinearity'  
def MG_V(st, xdelay, gamma, eta, p):
    return eta*(xdelay + gamma*st) / (1 + np.power((xdelay + gamma*st), p))

# Run signal transformation using Mackey-Glass oscillator reservoir + single
# dynamical node approach. s = input signal, mask = input mask, theta = virtual node
# spacing, tau = delay length, gamma = input scaling, eta = feedback strength, 
# p = nonlinearity
def RunMG_transform(s, mask, theta, tau, gamma, eta, p):
    # Define lengths for input/output matrices
    Ns = len(s)
    Nvirt = len(mask[0])

    # Mask input signal
    s_masked = np.zeros((Ns, Nvirt))
    for i in range(Ns):
        s_masked[i, :] = np.matmul(s[i], mask)

    # scale 'leak rate' from virtual node spacing
    alpha = np.exp(-theta)
    
    # Flatten to maintain timebase
    s_flat = s_masked.flatten()
    Xflat = np.zeros(Ns*Nvirt)
    
    # Convert tau to discrete time
    tau_d = int(tau/theta)
    
    # Transform input signal to output
    for t in range(1, Ns*Nvirt):
        Xflat[t] = Xflat[t-1] * alpha + (1 - alpha) * MG_V(s_flat[t], Xflat[t-tau_d], gamma, eta, p)
        
    # Reshape output
    Xout = np.reshape(Xflat, (Ns, Nvirt))
    return Xout

'''

Neural network functions

'''

# Calculate weights between reservoir states and target labels that minimise
# error with l2 penalty term (gamma)
def RidgeRegression(states, target, gamma, bias=True):
    # Ensure numpy
    if torch.is_tensor(states):
        states = states.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if bias==True:
        # Add bias to states
        bias = np.ones((len(states), 1))
        states = np.concatenate((bias, states), axis=1)
    # Setup matrices from inputs
    M1 = np.matmul(states.transpose(), target) 
    M2 = np.matmul(states.transpose(), states)
    # Perform ridge regression
    weights = np.matmul(np.linalg.pinv(M2+gamma*np.identity(len(M2))), M1)
    return weights
    
# Initialise linear readout. Nin = number of inputs, Nout = number of outputs          
def Linear_readout(Nin, Nout, actfn=nn.Softmax()):
    model = nn.Sequential(nn.Linear(Nin, Nout), actfn)
    model.train()
    return model

# Initialise 1 hidden layer mlp, Nh = number of hidden nodes        
def hidden_readout(Nin, Nh, Nout, actfn=nn.Softmax()):
    model = nn.Sequential(nn.Linear(Nin, Nh), nn.Tanh(), nn.Linear(Nh, Nout), actfn)
    model.train()
    return model
    
# Train model from provided input/output matrices. model = initialised linear readout,
# X = reservoir state data, Y = Target labels, (t = training, v = validation), Nbat = minibatch size, 
# epochs = number of training epochs, lr = learning rate, l2 = l2 penalty term, lossfn = loss function for GD. 
def online_training(model, Xt, Yt, Xv, Yv, Nbat, epochs, lr, l2, lossfn=nn.MSELoss(), bias=True, iter_per_return=1):
    # Initialise Adam Optimiser with l2 penalty term
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    # Initialise outputs for training/validation losses
    tloss = []
    vloss = []
    if bias==True:
        Xbias = torch.ones(len(Xt), 1)
        Xbiasv = torch.ones(len(Xv), 1)
        Xt = torch.cat((Xbias, Xt), axis=1)
        Xv = torch.cat((Xbiasv, Xv), axis=1)
    # Loop over number of epochs
    for i in range(epochs):
        # Randomly sample minibatch data from training/target data
        inds = torch.randint(0, len(Xt), [Nbat])
        Xin = Xt[inds]
        Yin = Yt[inds]       
        # Forward pass
        Ypred = model(Xin)
        # Calculate loss
        loss = lossfn(Ypred, Yin)
        # Reset gradients
        optim.zero_grad()
        # Backward pass
        loss.backward()
        # Update parameters
        optim.step()
        
        if i%iter_per_return == 0:
            # Report training loss
            tloss.append(loss.item())
            
            # Repeat on validation set
            Yvalid = model(Xv)
            loss_v = lossfn(Yvalid, Yv)
            vloss.append(loss_v.item())
        
        # Print update of loss every 1000 epochs
        if i%1000==0:
            print('Training loss: ', loss.item(), 'Validation loss: ', loss_v.item())
            
    return tloss, vloss
            
    
# Evaluates linear memory capacity for a system from a random input signal and the
# resulting reservoir states
def evaluate_linear_MC(signal, states, splits=[0.2, 0.8], delays=50):
    # ensure flat input signal
    signal = np.asarray(signal).flatten()
    # generate target signal from delayed input signal
    shift = np.zeros((len(signal), delays))
    for i in range(len(signal)-delays):
        i += delays
        shift[i, :] = signal[i-delays:i]
    # split data
    wash, Ytrain, Ytest = np.split(shift, [int(len(signal)*splits[0]), int(len(signal)*splits[1])])
    wash, Xtrain, Xtest = np.split(states, [int(len(signal)*splits[0]), int(len(signal)*splits[1])])
    # sweep over range of hyperparameters gamma to find optimal MC
    bestMC = 0
    gammas = np.logspace(-10, 0, 11)
    for gamma in gammas:
        # Calculate weights
        weights = RidgeRegression(Xtrain, Ytrain, gamma, bias=False)
        # Predict test states
        prediction = np.matmul(Xtest, weights)
        # Loop over all delays k and evaluate MC_k
        MC_k = np.zeros(delays)
        for k in range(delays):
            # Take prediction and target for each delay
            pred = prediction[:, k]
            targ = Ytest[:, k]
            # Set up matrix to calculate covariance
            M = pred, targ
            # Calculate covariance
            coVarM = np.cov(M)
            # Take cov(xy) 
            coVar = coVarM[0,1]
            # Measure the variance of the signals
            outVar = np.var(pred)
            targVar = np.var(targ)
            # Calculate the total variance of the raw target and the specific
            # target
            totVar = outVar*targVar
            # If the covariance coefficient is greater than 0.1, treat as better
            # than random guessing and add to MC_k outputs
            if coVar**2/totVar > 0.1:
                MC_k[k] = coVar**2/totVar
        # Account for floating point errors in MC
        MC_k[MC_k>1] = 1
        # Sum memory capacity over all delays
        MC = sum(MC_k)
        # If best reported MC, save data
        if MC > bestMC:
            bestMC = MC
    return bestMC
    


'''

Functions for loading datasets for ML tasks

'''

def load_EMNIST_data(tensor=True):
    data = np.load('EMNIST data.npy')
    labels = np.load('EMNIST labels.npy')
    if tensor == True:
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)
        data = data.double()
        labels = labels.double()
    return data, labels
    

'''

Functions for using SpaRCe

'''
# Initialise a thresholded readout layer. Xt, Yt = training data/labels, eta = learning rate (weights)
# beta = learning rate (thresholds), Pn = tuple of initialisation sparsities
def Initialise_sparce(Xt, Yt, batch_size, eta, beta, Pn):
    N = np.shape(Xt)[1]
    N_class = np.shape(Yt)[1]
    model = Classification_ReadOuts(N, N_class, batch_size, Pn)
    model.Initialise_SpaRCe(Xt, eta, beta, Pn)
    return model

# Online training of output weights and threshold values. model should be initialised sparce
def train_sparce(model, Xt, Yt, Xv, Yv, batch_size, eta, epochs, iter_per_return=1000):
    ACC = np.zeros([2, model.N_copies, epochs//iter_per_return])
    CL = np.zeros([2, model.N_copies, epochs//iter_per_return])
    for i in range(epochs):
        inds = np.random.randint(0, np.shape(Xt)[0], batch_size)
        state = torch.clone(Xt[inds, :])
        labels = torch.clone(Yt[inds, :])
        if i > 0:
            tpred, error, state_sparce = model.SpaRCe_Step(state, labels)
        if i % iter_per_return == 0:
            vstate = torch.clone(Xv)
            vlabels = torch.clone(Yv)
            vout, vacc, verr, vsp, vstsp = model.SpaRCe_Evaluate(vstate, vlabels)
            print('Epoch: '+str(i), 'Loss for Pn[0]: '+str(verr[0]))
            
# Evaluate testing accuracy
def classify_sparce(model, Xtest, Ytest):
    output, accuracy, error, sparsity, sparse_states = model.SpaRCe_Evaluate(Xtest, Ytest)
    out = []
    for i in accuracy:
        out.append(i.detach().numpy())
    out = np.asarray(out, dtype='float')
    return out


'''


Ring Array Simulations


'''
import torch
from NODE_Util import *

# Loads neural ODE + starting trajectory
# Load via model, start_traj = load_NODE(), adding keyword args if required for path changes/ cpu usage
def load_NODE(path='TrainedNODEmodel_sr32.pt', device='cuda'):
    device=device
    N_ODE = torch.load('TrainedNODEmodel_sr32.pt', map_location=torch.device(device))
    N_ODE.to(device)
    N_ODE.eval()
    start_traj = np.load('Start Traj.npy', allow_pickle=True)
    start_traj = torch.from_numpy(start_traj)
    return N_ODE, start_traj

# Runs signal transformation via a model of nanoring arrays. model = loaded NODE,
# inputsignal = input data to be transformed, Hc/Hr = field scaling parameters, 
# start_traj = starting trajectories for the NODE
# Model only accurate for around 20 < HC < 35 and 1 < Hr < 25, as this is what the 
# NODE was trained on
def RunRingTransform(model, inputsignal, Hc, Hr, start_traj, device='cuda'):
    # Generate model input/output signals from input data:
        
    # Get shape of input signal
    inshape = np.shape(inputsignal)
    # Calculate number of batches needed for simulation
    Nbat = int(np.ceil(inshape[0]/50))
    # Generate blank matrix for inputs
    scaled_inputs = np.zeros((Nbat*50, inshape[1]))
    NODE_inputsignals = torch.zeros(Nbat*50, inshape[1]*32)
    NODE_outputsignals = torch.clone(NODE_inputsignals)
    # Scale inputs via Hc and Hr
    scaled_inputs[:inshape[0]] = (Hc + inputsignal*Hr)/36.75
    # Generate sinusoid
    sinot = np.sin(np.linspace(0, 2*np.pi-(2*np.pi/32), 32))
    # Tile sinusoid 
    tilesinot = np.tile(sinot, (Nbat*50, 1))
    # Turn static inputs into modulated sinusoids
    for i in range(inshape[1]):
        NODE_inputsignals[:, i*32:(i+1)*32] =  torch.from_numpy(scaled_inputs[:, i, None] * tilesinot)
    # Tile input signals to 3D signal
    NODE_inputsignals = torch.tile(NODE_inputsignals, [3, 1, 1])
    NODE_inputsignals = NODE_inputsignals.permute(1, 0, 2)
    
    # Perform neural ODE simulation
    # Set initial conditions
    t0=torch.tensor(0.).to(device)
    X0=torch.tensor(0.).to(device)
    # Loop over batches
    for k in range(Nbat):
        # Reset to initial conditions
        model.Reset(t0, X0)
        for t in range(inshape[1]*32):
            if t == 0:
                states = start_traj.to(device)
                out = model.ODE_forward_noerr(states,t0,Reset=False)
                NODE_outputsignals[k*50:(k+1)*50, t] = out[:, 0, :].flatten()
            else:
                states = torch.cat((NODE_inputsignals[k*50:(k+1)*50, :, t], out[:, :, 0]), 1).unsqueeze(2).to(device)
                out = model.ODE_forward_noerr(states,t0,Reset=False)
                NODE_outputsignals[k*50:(k+1)*50, t] = out[:, 0, :].flatten()
    return NODE_outputsignals
    
    
