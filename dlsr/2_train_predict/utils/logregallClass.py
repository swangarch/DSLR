from utils.logregClass import logreg, save_weights
from utils.train_utils import count_correct
import numpy as np
import pandas as pd
from datetime import datetime
import os


class logregall:
    """logistic regression for training all features"""


    def __init__(self, df, feature_names, classname, goals):
        """Init logistic regression for all features"""

        self.df = df
        self.feature_names = feature_names
        self.lgs = []
        self.goals = goals
        self.classname = classname
        for goal in self.goals:
            self.lgs.append(logreg(self.df, self.feature_names, self.classname, goal))
        
        self.weights = []
        print("[initialization done]\n")


    def train_all(self, learning_rate=0.00005, max_iter=10000, Debug=False):
        """Train models for all class, save weights for classifcation."""

        startTime = datetime.now()
        for i, goal in enumerate(self.goals):
            weights = self.lgs[i].train(learning_rate, max_iter, Debug)
            self.weights.append(weights.flatten())

        print(f"\033[031m[TOTAL TRAINING TIME] {datetime.now() - startTime}\033[0m")
        os.makedirs("output", exist_ok=True)
        save_weights(np.array(self.weights), "output/weights.csv")
        print("[weghts.csv save at output/weights.csv]")
        

    def predict(self, df_test): #### issue
        """Predict for test dataset."""

        if len(df_test) == 0:
            print("No validation data, skip validation.")
            return
        if len(self.weights) == 0:
            raise ValueError("No model weights yet.")
        predictions = [self.lgs[i].predict(df_test).flatten() for i in range(len(self.lgs))]

        index = np.argmax(np.stack(predictions), axis=0)
        final = np.array([self.goals[i] for i in index])
        truth = np.array(df_test[self.classname])
        print(count_correct("TEST FINAL", final, truth, False))


    def predict_new(self, df_new):
        """Predict for a new dataset."""

        if len(df_new) == 0:
            print("No validation data, skip validation.")
            return
        if len(self.weights) == 0:
            raise ValueError("No model weights yet.")
        predictions = [self.lgs[i].predict_new(df_new, self.weights[i]).flatten() for i in range(len(self.lgs))]
        index = np.argmax(np.stack(predictions), axis=0)

        final = [[count, self.goals[i]] for count, i in enumerate(index)]
        df = pd.DataFrame(final, columns=["Index", "Hogwarts House"])
        os.makedirs("output", exist_ok=True)
        df.to_csv("output/house.csv", index=False)
        print("[Prediction saved at output/house.csv]")


    def load_weights(self, path):
        """Load weights from file."""

        self.weights = np.loadtxt(path, delimiter=",", dtype=float).tolist()
        print("[WEIGHTS LOADED]")
        print(self.weights)
