#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb;
import time
import copy
from numpy import cos,sin,pi

from my_lib import *

get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')


# In[2]:


actuation_noise_std = np.ones((2,))*0.05*60

measurement_noise_std = np.array([0.003,0.003,4.98e-5,1.25e-5]) #np.ones((4,))*1e-5

measure_noise_cov = np.diag(measurement_noise_std)
state_noise_cov = np.diag(measurement_noise_std)


# In[3]:


def calc_mse(state_seq,pred_states2):
    m_state_seq = np.array(state_seq)
    m_pred_states2 =np.array(pred_states2)
#     print(m_pred_states2[:,0:3].shape)
    mse = np.sqrt(np.mean( (m_pred_states2[:,0:3] - m_state_seq[:,0:3,])**2,axis = 0))
    return mse
    

def eval_one_traj(control_seq,init_state, seed =0, plot = True):
    np.random.seed(seed)
    state_seq,obs_seq = trace_traj(init_state,control_seq,actuation_noise_std,measurement_noise_std)
    


    init_cov = np.zeros((4,4))
    actuation_noise_cov = np.diag(actuation_noise_std)**2
    measure_noise_cov = np.diag(measurement_noise_std)**2


    pred_states2 = apply_kalman2(obs_seq,control_seq,init_state,init_cov,actuation_noise_cov,measure_noise_cov,spreading = 1)
   
    if plot:
        plot_state_seq(state_seq,10)
        plot_state_seq(pred_states2,10)
        eval_states(pred_states2,state_seq)
        print("MSE" , calc_mse(state_seq,pred_states2))
    return (state_seq,pred_states2)

control_seq = [[50,60]]*200
init_state = [0.4,0.4,np.pi/2,0]
eval_one_traj(control_seq,init_state,seed = 0);


# In[4]:


def eval_batch_traj(control_seq,init_state, n_traj = 100, max_attempts = 500 ,seed =0 ):
    np.random.seed(seed)
    
    
    init_cov = np.zeros((4,4))
    actuation_noise_cov = np.diag(actuation_noise_std)**2
    measure_noise_cov = np.diag(measurement_noise_std)**2
    
    results = np.zeros((3,n_traj))
    
    correct_count = 0
    i = 0
    while (correct_count< n_traj):
        try:
            state_seq,obs_seq = trace_traj(init_state,control_seq,actuation_noise_std,measurement_noise_std)
            correct_count+=1
            pred_states2 = apply_kalman2(obs_seq,control_seq,init_state,init_cov,actuation_noise_cov,measure_noise_cov,spreading = 1)
            mse = calc_mse(state_seq,pred_states2)
#             print(mse)
            results[:,correct_count] = mse
        except:
            pass
        i=i+1
        if i>= max_attempts:
            1/0
            break;
        
    mn_results = np.mean(results,axis = 1)
    std_results = np.std(results,axis = 1)
    print(mn_results,std_results)
    
    
    return (mn_results,std_results)

control_seq = [[50,60]]*200
init_state = [0.4,0.4,np.pi/2,0]
eval_batch_traj(control_seq,init_state)


# In[5]:


traj1 = [[50,50]]*50
init_state1 = [0.25,0.1, 0.0, 0] 
eval_one_traj(traj1,init_state1,seed = 9);


# In[6]:


n_traj = 500
max_attempts = 2000


# In[7]:


x_vals = [0.15,0.35]
x_mn = np.zeros((len(x_vals),3))
x_std = np.zeros((len(x_vals),3))


control_seq = [[60,60]]*100
eval_one_traj(control_seq,[0.3,0.1,0,0] )
for i in range(len(x_vals)):
    print(i)
    x = x_vals[i]
    init_state = [x,0.1,0,0]
    [x_mn[i],x_std[i]] = eval_batch_traj(control_seq,init_state,n_traj=n_traj,max_attempts = max_attempts)


# In[8]:


plt.errorbar(x_vals,x_mn[:,0],yerr = x_std[:,0],  uplims=True, lolims=True)
plt.errorbar(x_vals,x_mn[:,1],yerr = x_std[:,1],  uplims=True, lolims=True)
plt.errorbar(x_vals,x_mn[:,2],yerr = x_std[:,2],  uplims=True, lolims=True)
plt.legend(['x','y','th'])


# In[ ]:


y_vals = [0.2,0.5]
y_mn = np.zeros((len(y_vals),3))
y_std = np.zeros((len(y_vals),3))


control_seq = [[60,60]]*100
eval_one_traj(control_seq,[0.45,0.35,np.pi/2,0] )
for i in range(len(y_vals)):
    print(i)
    y = y_vals[i]
    init_state = [0.05,y,np.pi/2,0]
    [y_mn[i],y_std[i]] = eval_batch_traj(control_seq,init_state,n_traj=n_traj,max_attempts = max_attempts)


# In[ ]:


plt.errorbar(y_vals,y_mn[:,0],yerr = y_std[:,0],  uplims=True, lolims=True)
plt.errorbar(y_vals,y_mn[:,1],yerr = y_std[:,1],  uplims=True, lolims=True)
plt.errorbar(y_vals,y_mn[:,2],yerr = y_std[:,2],  uplims=True, lolims=True)
plt.legend(['x','y','th'])


# In[ ]:


for _ in range(100):
    print()

