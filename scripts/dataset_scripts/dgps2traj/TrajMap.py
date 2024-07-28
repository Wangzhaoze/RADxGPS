import numpy as np
import pandas as pd
import os
from matplotlib import cm
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class TrajGridMap:
    def __init__(self):
        
        self.MapShape = (1000, 1000)
        self.Map = np.zeros(self.MapShape)
        self.sigma = 150 #6sigma = 4.8m = 48pixels
        pass

    def map_clear(self):
        self.Map = np.zeros(self.MapShape)
        pass

    def RowCol2XY(self, Idx_row, Idx_col):
        X = Idx_col
        Y = 1000 - Idx_row
        return X, Y

    def XY2RowCol(self, X, Y):
        Idx_col = X
        Idx_row = 1000 - Y
        return Idx_row, Idx_col

    def gaussian1D(self, Idx_row, Idx_col):
        MapLength, MapWidth = self.MapShape
        x = np.arange(0, MapWidth)
        new_map = np.zeros(self.MapShape)
        for i, col in enumerate(Idx_col):
            mean = col
            gaussian = multivariate_normal(mean = mean, cov = self.sigma)
            z = gaussian.pdf(x)
            new_map[Idx_row[i], :] = z / np.max(z)
        return new_map


    def map_update(self, Idx_row, Idx_col):
        # make sure map clear
        self.map_clear()

        # filter map limit
        #Idx_row, Idx_col = self.map_limit_filter(Idx_row, Idx_col)

        self.Map = self.gaussian1D(Idx_row, Idx_col)



    














