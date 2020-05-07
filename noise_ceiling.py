# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:12:05 2020

@author: cuijy
"""

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

y_res = pd.read_csv('y_res1.csv', header=None)

col_names=['x', 'y', 'z', 'depth']
location=pd.read_csv('location.csv',names=col_names)


#define a function get corr from any given y-res
def get_corr(Y_res):
    
    #obtaining the correlation between voxels
    corr=Y_res.corr(method ='pearson')
    corr_tril=np.tril(corr,k=-1)
    Corr=corr_tril[np.nonzero(corr_tril)]
    Corr=Corr.reshape(-1,1)
    Corr=pd.DataFrame(Corr)
    return Corr


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
    
    design_dict={"exp": X_exp,
                "power": X_power}
    return design_dict


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
        
