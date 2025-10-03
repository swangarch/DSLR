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
        for i, goal in enumerate(self.goals):
            self.lgs.append(logreg(self.df, self.feature_names, self.classname, goal))
        
        self.weights = []
        print("[initialization done]\n")

    def train_all(self, learning_rate=0.00005, max_iter=10000, Debug=False):
        """Train models for all class, save weights for classifcation."""

        startTime = datetime.now()
        for i, goal in enumerate(self.goals):
            weights = self.lgs[i].train(learning_rate, max_iter, Debug)
            self.weights.append(weights)

        print(f"\033[031m[TOTAL TRAINING TIME] {datetime.now() - startTime}\033[0m")
        os.makedirs("output", exist_ok=True)
        save_weights(np.array(self.weights), "output/weights.csv")
        
    def predict_test(self, df_test):
        """Predict for test dataset."""

        if len(df_test) == 0:
            print("No validation data, skip validation.")
            return
        if len(self.weights) == 0:
            raise("No model weights yet.")
        predictions = []
        for i in range(len(self.lgs)):
            predictions.append(self.lgs[i].predict(df_test))
        final_prediction = np.stack(predictions).transpose()

        final_index = np.zeros(len(final_prediction), dtype=int)
        for i, arr in enumerate(final_prediction):
            max_idx = np.argmax(arr)
            final_index[i] = max_idx

        final = [self.goals[i] for i in final_index]
        truth = list(df_test[self.classname])

        print(count_correct("TEST FINAL", final, truth))

    def predict_new(self, df_new):
        """Predict for a new dataset."""

        if len(self.weights) == 0:
            raise("No model weights yet.")
        if len(df_new) == 0:
            raise("No data.")
        
        predictions = []
        for i in range(len(self.lgs)):	
            predictions.append(self.lgs[i].predict_new(df_new, self.weights[i]))
        final_prediction = np.stack(predictions).transpose()

        final_index = np.zeros(len(final_prediction), dtype=int)
        for i, arr in enumerate(final_prediction):
            max_idx = np.argmax(arr)
            final_index[i] = max_idx

        print("\033[33m[FINAL PREDICTION SAVED] house.csv\33[0m]")

        final = []
        for count, index in enumerate(final_index):
            final.append([count, self.goals[index]]) 
        df = pd.DataFrame(final, columns=["Index", "Hogwarts House"])
        os.makedirs("output", exist_ok=True)
        df.to_csv("output/house.csv", index=False)

    def load_weights(self, path):
        """Load weights from file."""

        self.weights = np.loadtxt(path, delimiter=",", dtype=float).tolist()
        print("[WEIGHTS LOADED]")
        print(self.weights)
