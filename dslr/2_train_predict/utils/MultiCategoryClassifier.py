from utils.logregClass import Logreg
from utils.train_utils import count_correct, save_weights
import numpy as np
import pandas as pd
from datetime import datetime
import os


class MultiCategoryClassifier:
    """logistic regression for training all features"""

    def __init__(self, df, feature_names, classname, goals):
        """Init logistic regression for all features"""

        self.df = df
        self.feature_names = feature_names
        self.lgs = []
        self.goals = goals
        self.classname = classname
        for goal in self.goals:
            self.lgs.append(Logreg(self.df, self.feature_names, self.classname, goal))
        
        self.weights = []
        print("[Initialization done]\n")

    def train_all(self, learning_rate:float=0.00005, max_iter:int=10000, batch_size:int=100) -> None:
        """Train models for all class, save weights for classifcation."""

        if learning_rate <= 0 or max_iter <= 0 or batch_size < 1:
            raise ValueError("Wrong training parameters.")
        print("[Training start] =>", "<SGD>" if batch_size == 1 else "<mini batch> " + str(batch_size))
        startTime = datetime.now()
        for i, goal in enumerate(self.goals):
            weights = self.lgs[i].train(learning_rate, max_iter, batch_size)
            self.weights.append(weights.flatten())

        print(f"\033[032m[TOTAL training time] {datetime.now() - startTime}\033[0m")
        os.makedirs("output", exist_ok=True)
        save_weights(np.array(self.weights), "output/weights.csv")
        print("[weghts.csv save at output/weights.csv]")

    def predict(self, df_test:pd.DataFrame) -> None:
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
        print(count_correct("[Test final]", final, truth, False))

    def predict_new(self, df_new:pd.DataFrame) -> None:
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
        df.to_csv("output/houses.csv", index=False)
        print("[Prediction saved at output/houses.csv]")

    def load_weights(self, path:str) -> None:
        """Load weights from file."""

        self.weights = np.loadtxt(path, delimiter=",", dtype=float).tolist()
        print("[Weights loaded]")

