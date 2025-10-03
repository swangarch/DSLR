#!/usr/bin/python3

import numpy as np
from numpy import ndarray as array
from utils.train_utils import normalize, prob_predict, preproc_onevsall, cal_grad, count_correct
from datetime import datetime
from pandas import DataFrame as dataframe


class logreg:
    """Class to perform logistic regression."""

    def __init__(self, data: dataframe, feature_names:list, className: str, goal: str):
        """Init logreg."""

        self.classname = className
        self.goal = goal
        catArr = np.array(data[className])
        self.features = []
        self.feature_names = feature_names
        for name in feature_names: # add check if length matched
            self.features.append(np.array(data[name]))
        self.catToTrain = preproc_onevsall(catArr, goal)
        self.length = len(self.catToTrain)
        self.ranges = [] #tuple list
        # self.weights = [0.0] #scalar list
        self.weights = np.zeros(len(self.feature_names) + 1, dtype=np.float32)
        self.fnorms = [] #array list

        for i, f in enumerate(self.features):
            self.ranges.append((float(min(f)), float(max(f)))) ##can we use?? min max save the range
            self.fnorms.append(normalize(f, self.ranges[i]))

    def train(self, learning_rate=0.00005, max_iter=10000, Debug=False) -> None:
        """Train algorithm for a feature."""

        startTime = datetime.now()
        for i in range(max_iter):
            res = prob_predict(self.weights, self.fnorms, self.length)
            gradient = cal_grad(res, self.catToTrain, self.fnorms, self.weights)
            self.weights -= gradient * learning_rate #gradient descent

            if i % 10 == 0:
                binary_arr = (res > 0.5).astype(int)
                print("\033[?25l\033[033m[ITER]", int(i), count_correct(self.goal + " TRAIN", binary_arr, self.catToTrain), "\033[0m", end='\r')
                if Debug:
                    print("\n")
                self.debug_info(Debug, gradient, res)

        print(f"\033[?25h[----------------------{self.goal} training done----------------------------]")
        print(f"\033[031m[{self.goal} TRAINING TIME] {datetime.now() - startTime}\033[0m\n")
        return self.weights

    def predict(self, df_test: dataframe)-> array:
        """Predict after training then compare with ground truth."""
        
        weights = self.weights
        ranges = self.ranges
        fnorms = []
        features = []
        for name in self.feature_names: # add check if length matched
            features.append(np.array(df_test[name]))
        for i, f in enumerate(features):
            fnorms.append(normalize(f, ranges[i]))
        prob = prob_predict(weights, fnorms, len(df_test))
        return prob
    
    def predict_new(self, df_test: dataframe, weights: array)-> array:
        """Predict after training then compare with ground truth."""
        
        ranges = self.ranges
        fnorms = []
        features = []
        for name in self.feature_names: # add check if length matched
            features.append(np.array(df_test[name]))
        for i, f in enumerate(features):
            fnorms.append(normalize(f, ranges[i]))
        length = len(df_test)
        prob = prob_predict(weights, fnorms, length)
        return prob
    
    def debug_info(self, debug_enalbled:bool, gradient: array, res:array) -> None:
        if debug_enalbled:
            print("\033[033m----------------------------------------------")
            print("\033[034m[GRAD]", gradient, "\033[0m")
            print("\033[035m[WEIS]", self.weights, "\033[0m")
            print("\033[032m[RESS]", res, "\033[0m")
            print("----------------------------------------------\033[0m")


def save_weights(weight: array, path:str) -> None:
    """Save the weights."""

    try:
        np.savetxt(path, weight, delimiter=",", fmt="%f")
    
    except Exception as e:
        print("Error:", e)

