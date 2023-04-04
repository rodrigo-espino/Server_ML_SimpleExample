#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:23:19 2023

@author: rodrigosantacruzespino
"""
# Import Libraries#
import pandas as pd
import joblib

#Creating Class Utils 
class Utils:
    #
    # Creating function to return a pandas DataFrame from CSV#
    def load_from_csv(self, path):
        return pd.read_csv(path)
    #
    # Creating function for setting X and Y for our model #
    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X, y
    
    #
    # Creating function for exporting our model #
    def model_export(self, clf, score):
        joblib.dump(clf, './models/best_model.pkl')