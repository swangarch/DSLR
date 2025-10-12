#!/usr/bin/python3

import numpy as np
from numpy import ndarray as array
from utils.train_utils import normalize, prob_predict, preproc_onevsall, cal_grad, count_correct, cross_entropy_loss
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
        self.weights = np.zeros((1, len(self.feature_names) + 1), dtype=np.float32)
        fnorms = [np.ones((1, len(catArr)))] #array list

        for i, f in enumerate(self.features):
            self.ranges.append((float(min(f)), float(max(f)))) ##can we use?? min max save the rang
            fnorms.append(normalize(f, self.ranges[i]))
        
        self.fnorms = np.vstack(fnorms)


    def train(self, learning_rate=0.00005, max_iter=10000, Debug=False, batch_size=50) -> None:
        """Train algorithm for a feature."""

        startTime = datetime.now()
        for i in range(max_iter):
            # count = 0
            # while count <= batch_size:
            res = prob_predict(self.weights, self.fnorms, self.length)
            gradient = cal_grad(res, self.catToTrain, self.fnorms, self.weights)
            self.weights -= gradient * learning_rate #gradient descent
            # count += batch_size

            if i % 10 == 0:
                binary_arr = (res > 0.5).astype(int)
                correct_num = count_correct(self.goal + " TRAIN", binary_arr, self.catToTrain, True)
                print("\033[?25l\033[033m[ITER]", int(i), correct_num, "\033[0m", end='\r')
                if Debug:
                    print("\n")
                self.debug_info(Debug, gradient, res)

        print()
        print(f"\033[?25h[{self.goal} training done]")
        print(f"\033[031m[{self.goal} TRAINING TIME] {datetime.now() - startTime}\033[0m\n")
        return self.weights


    def predict(self, df_test: dataframe)-> array:
        """Predict after training then compare with ground truth."""
        
        weights = self.weights
        ranges = self.ranges
        l = len(df_test)
        ranges.insert(0, (0,1))
        fnorms = []
        features = [np.ones((1, l))]
        for name in self.feature_names: # add check if length matched
            features.append(np.array(df_test[name]))
        features = np.vstack(features)
        for i, f in enumerate(features):
            fnorms.append(normalize(f, ranges[i]))
        fnorms = np.array(fnorms)
        prob = prob_predict(weights, fnorms, len(df_test))
        return prob
    

    def predict_new(self, df_test: dataframe, weights: array)-> array:
        """Predict after training then compare with ground truth."""
        
        ranges = self.ranges
        length = len(df_test)
        fnorms = [np.ones((1, length))]
        features = []
        for name in self.feature_names: # add check if length matched
            features.append(np.array(df_test[name]))
        for i, f in enumerate(features):
            fnorms.append(normalize(f, ranges[i]))
        
        arr_w = np.array(weights).reshape(1, -1)
        arr_f = np.vstack(fnorms)
        prob = prob_predict(arr_w, arr_f, length)
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

