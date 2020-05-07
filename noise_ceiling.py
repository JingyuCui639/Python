#!/usr/bin/env python
# coding: utf-8

# # Noise ceiling function

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
import scipy.spatial as sp
import scipy.optimize as so
import random

get_ipython().run_line_magic('matplotlib', 'inline')


# # Import datasets

# In[15]:


y_res = pd.read_csv('y_res1.csv', header=None)

col_names=['x', 'y', 'z', 'depth']
location=pd.read_csv('location.csv',names=col_names)


# # View y_res.csv

# In[16]:


print(y_res.shape)
y_res.head()


# # View location.csv

# In[17]:


print(location.shape)
location.head()


# # Define a function get correlation from any given y_res

# In[18]:


def get_corr(Y_res):
    
    #obtaining the correlation between voxels
    corr=Y_res.corr(method ='pearson')
    corr_tril=np.tril(corr,k=-1)
    Corr=corr_tril[np.nonzero(corr_tril)]
    Corr=Corr.reshape(-1,1)
    Corr=pd.DataFrame(Corr)
    return Corr


# # Define a function get design matrix X from dataframe: location
# ### exp_model: $Corr=\beta_0+\beta_1*exp(-dist*\gamma)+\beta_2*dep1+\beta_3*dep2+\beta4*dep1*dep2$
# ### power_model: $Corr=\beta_0+\beta_1*eta^{dist}+\beta_2*dep1+\beta_3*dep2+\beta4*dep1*dep2$

# In[19]:


def get_design_matrix(Location,gamma=0.2,eta=0.8):

    #Obtaining the distance between voxels
    dist_matrix=sp.distance_matrix(Location.iloc[:,0:2],Location.iloc[:,0:2])
    dist_tril=np.tril(dist_matrix,k=-1)
    rowid,colid=np.nonzero(dist_tril)
    dist=dist_tril[rowid,colid]
    dist=dist.reshape(-1,1)
    
    exp_dist=np.exp(-dist*gamma)
    power_dist=[eta**k for k in dist]

    #Obtaining the depth from the 1st voxel
    dep1=Location.depth[rowid]
    dep1=dep1.values
    dep1=dep1.reshape(-1,1)

    #obtaining depth from the 2ed voxel
    dep2=Location.depth[colid]
    dep2=dep2.values
    dep2=dep2.reshape(-1,1)

    dep1_2=dep1*dep2

    #Creat the design matrix
    X_exp=np.concatenate((exp_dist,dep1,dep2,dep1_2),axis=1)
    names_exp=['exp_dist','dep1','dep2','dep12']
    X_exp=pd.DataFrame(X_exp,columns=names_exp)
    
    X_power=np.concatenate((power_dist,dep1,dep2,dep1_2),axis=1)
    names_power=['power_dist','dep1','dep2','dep12']
    X_power=pd.DataFrame(X_power,columns=names_power)
    
    #make design matrices as a dictionary
    design_dict={"exp": X_exp,
                "power": X_power}
    
    #return a dictionary with model names and corresponding design matrix X
    return design_dict


# ## Function model_evaluation
# ### taking input: noise matrix, model (design matrix X), evaluation method , num of runs; 
# ### returning: the $R^2$ for each of the model,  lower bound  and upper bound of noise ceiling

# In[20]:


def model_evaluation(Y_res, design_matrix=None, evaluation_method="R_sqr",num_run=8):
    #design_matrix: a dictionary of a list of design matrices (with name as index name)

    #split Y_res into n equal pieces according to the number of runs (n)
    
    r_sqr_low=np.zeros(num_run)
    r_sqr_up=np.zeros(num_run)
    kf = KFold(n_splits=8)
    i=0
    for train_index, test_index in kf.split(Y_res):
        
        Corr_train=get_corr(Y_res.loc[train_index])
        Corr_test=get_corr(Y_res.loc[test_index])
        Corr_allrun=get_corr(Y_res)
        
        ESS_lowbound=np.sum((Corr_train-Corr_test)**2)
        ESS_upbound=np.sum((Corr_allrun-Corr_test)**2)
        mu=np.mean(Corr_test)
        TSS=np.sum((Corr_test-mu)**2)
        
        if evaluation_method=="R_sqr":
            
            r_sqr_low[i]=1-ESS_lowbound/TSS
            r_sqr_up[i]=1-ESS_upbound/TSS
               
    R_sqr_lowbound=np.mean(r_sqr_low)
    R_sqr_upbound=np.mean(r_sqr_up)
    noise_ceilling_bounds=[R_sqr_lowbound,R_sqr_upbound]    

    r_sqr_model={}
    if design_matrix!=None:       
      
        for model_key in design_matrix:
            linmodel = LinearRegression().fit(design_matrix[model_key], Corr_allrun)
            r_sqr_model[model_key]=linmodel.score(design_matrix[model_key], Corr_allrun)
            
    return [r_sqr_model,noise_ceilling_bounds]


# ### Test function

# In[21]:


model_dict=get_design_matrix(location)
results_model_evaluat=model_evaluation(y_res, model_dict)
results_model_evaluat


# ## Creating a graph showing the result from function: model_evaluation()

# In[22]:


def barplot_noisebounds(result_model_evaluation, plot_type="bar"):
    results_dist=result_model_evaluation[0]
    model_name=[key for key in results_dist]
    r_sqr=[results_dist[key] for key in results_dist]
    
    lowbound, upbound=result_model_evaluation[1]
    
    if plot_type=="bar":
        plt.figure(figsize=(10,6))
        ax=sns.barplot(model_name,r_sqr)
        #change the bar width 
        widthbars=[0.3,0.3]
        for bar ,newwidth in zip(ax.patches, widthbars):
            x=bar.get_x()
            width=bar.get_width()
            center=x+width/2
            bar.set_x(center-newwidth/2.)
            bar.set_width(newwidth)
        #ax.set(ylim=(0, 0.2))
        ax.axhline(lowbound, ls='--',color="gray")
        ax.axhline(upbound, ls='--',color="gray")


# In[23]:


barplot_noisebounds(results_model_evaluat)

