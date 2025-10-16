#!/usr/bin/python3

import numpy as np
from numpy import ndarray as array
from utils.train_utils import normalize, prob_predict, preproc_onevsall, cal_grad, count_correct, cross_entropy_loss
from datetime import datetime
from pandas import DataFrame as dataframe
import sys


class Logreg:
    """Class to perform logistic regression."""

    def __init__(self, data: dataframe, feature_names:list, className: str, goal: str):
        """Init logreg."""

        self.classname = className
        self.goal = goal
        catArr = np.array(data[className])
        self.features = []
        self.feature_names = feature_names
        for name in feature_names:
            self.features.append(np.array(data[name]))
        self.catToTrain = preproc_onevsall(catArr, goal)
        self.length = len(self.catToTrain)
        self.ranges = [] #tuple list
        self.weights = np.zeros((1, len(self.feature_names) + 1), dtype=np.float32)
        fnorms = [np.ones((1, len(catArr)))] #array list

        for i, f in enumerate(self.features):
            self.ranges.append((float(min(f)), float(max(f))))
            fnorms.append(normalize(f, self.ranges[i]))
        
        self.fnorms = np.vstack(fnorms)


    def train(self, learning_rate=0.00005, max_iter=5000, batch_size=50) -> None:
        """Train algorithm for a feature."""

        startTime = datetime.now()
        l = self.fnorms.shape[1]
        truth = self.catToTrain.reshape(1, -1)
        for i in range(max_iter):
            count = 0
            while True:
                batch_input = self.fnorms[:, count: count + batch_size]
                batch_truth = truth[:, count: count + batch_size]
                prediction = prob_predict(self.weights, batch_input)
                gradient = cal_grad(prediction, batch_truth, batch_input, self.weights)
                self.weights -= gradient * learning_rate #gradient descent
                count += batch_size
                if count > l:
                    break

            if i % 50 == 0:
                p = prob_predict(self.weights, self.fnorms)
                binary_arr = (p > 0.5).astype(int)
                correct_num = count_correct(self.goal + " TRAIN", binary_arr, truth, True)
                print("\033[?25l\033[033m[Epoch]", int(i), correct_num, "\033[0m", end='\r')

        print()
        print(f"\033[?25h[{self.goal} training done]   [time] {datetime.now() - startTime}\n")
        return self.weights

    def predict(self, df_test: dataframe)-> array:
        """Predict after training then compare with ground truth."""
        
        weights = self.weights
        ranges = self.ranges
        l = len(df_test)
        ranges.insert(0, (0,1))
        fnorms = []
        features = [np.ones((1, l))]
        for name in self.feature_names:
            features.append(np.array(df_test[name]))
        features = np.vstack(features)
        for i, f in enumerate(features):
            fnorms.append(normalize(f, ranges[i]))
        fnorms = np.array(fnorms)
        prob = prob_predict(weights, fnorms)
        return prob

    def predict_new(self, df_test: dataframe, weights: array)-> array:
        """Predict after training then compare with ground truth."""
        
        ranges = self.ranges
        fnorms = [np.ones((1, len(df_test)))]
        features = []
        for name in self.feature_names:
            features.append(np.array(df_test[name]))
        for i, f in enumerate(features):
            fnorms.append(normalize(f, ranges[i]))
        
        arr_w = np.array(weights).reshape(1, -1)
        arr_f = np.vstack(fnorms)
        prob = prob_predict(arr_w, arr_f)
        return prob



