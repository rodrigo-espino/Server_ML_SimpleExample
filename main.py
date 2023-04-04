#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:20:16 2023

@author: rodrigosantacruzespino
"""

# Import Libraries#
from utils import Utils
from models import Models

# Creating main function
if __name__ == "__main__":
    utils = Utils()
    models = Models()
    
    # Loading data from CSV
    data = utils.load_from_csv('./in/felicidad.csv')
    # setting the value for X and Y
    X, y = utils.features_target(data, ['score', 'rank', 'country'], ['score'])
    # Training the model and verify which is the best model
    models.grid_training(X, y)