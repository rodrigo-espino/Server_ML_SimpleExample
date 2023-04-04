#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:23:51 2023

@author: rodrigosantacruzespino
"""
# Import Libraries#

import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from utils import Utils # Importing Utils class from utils.py

class Models:
    def __init__(self):
    #Creating a dictionary with the models we want to use
        self.reg = {
            'SVR': SVR(),
            'GRADIENT': GradientBoostingRegressor()
        }
        #Creating a dictionary with the parameters we want to use
        self.params = {
            'SVR' : {
                'kernel': ['linear', 'poly', 'rbf'],
                'gamma': ['auto', 'scale'],
                'C': [1,5,10]
            }, 'GRADIENT':{
                'loss': ['ls', 'lad'],
                'learning_rate': [0.01, 0.05, 0.1]
            }    
        }
        
    def grid_training(self, X, y):
        #Creating a variable to store the best score and model
        best_score = 999
        best_model = None
        
        #Creating a loop to iterate over the models and parameters
        for name, reg in self.reg.items():
            #Creating a GridSearchCV object
            grid_reg = GridSearchCV(reg, self.params[name], cv=3).fit(X, y.values.ravel())
            #Getting the best score
            score = np.abs(grid_reg.best_score_)
            #Checking if the score is better than the previous one
            if score < best_score:
                #If it is, we store the model and the score
                best_model = grid_reg.best_estimator_
                #Printing the best score and the best model
                best_score = score
       #Returning the best model and the best score         
        utils = Utils()
        #Calling the model_export function from utils.py
        utils.model_export(best_model, best_score)
            