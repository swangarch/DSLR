#!/usr/bin/python3

from utils.load_csv import load
import pandas as pd
import sys
from utils.MultiCategoryClassifier import MultiCategoryClassifier
from utils.train_utils import clean_data


def main():
	"""Main to load dataset and train."""

	try:
		print("\033[33mUsage: python3 logreg_predict.py <train_set_csv> <test_set_csv> <weights.csv>\033[0m")
		argv = sys.argv
		assert len(argv) == 4, "Wrong argument number."
		pd.set_option('display.float_format', '{:.6f}'.format)

		df = load(argv[1])
		df_new = load(argv[2])
		df_new = clean_data(df_new, "mean")
		
		feature_names = [
							"Astronomy", "Herbology", 
							"Arithmancy", 
							"Charms", 
							"Divination", "Ancient Runes",
							"Defense Against the Dark Arts",
							"Muggle Studies", "History of Magic",
							"Transfiguration", 
							"Potions", "Care of Magical Creatures",
							"Flying",
						]

		classname = "Hogwarts House"
		goals = ["Ravenclaw", "Gryffindor", "Slytherin", "Hufflepuff"]

		mcc = MultiCategoryClassifier(df, feature_names, classname, goals)
		mcc.load_weights(argv[3])
		mcc.predict_new(df_new)

	except KeyboardInterrupt:
		print("\033[33mStopped by user.\033[0m")

	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()
