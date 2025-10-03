#!/usr/bin/python3

from utils.load_csv import load
import pandas as pd
import sys
from utils.logregallClass import logregall
from utils.train_utils import clean_data, split_data


def main():
	"""Main to load dataset and train."""

	try:
		print("\033[33mUsage: python3 logreg_train.py <path_csv>\033[0m")
		argv = sys.argv
		assert len(argv) == 2, "Wrong argument number."
		pd.set_option('display.float_format', '{:.6f}'.format)

		df = load(argv[1])
		df = clean_data(df, "mean")
		df, df_test = split_data(df, 0.8, 4123232)
		
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

		lgall = logregall(df, feature_names, classname, goals)
		lgall.train_all()
		lgall.predict_test(df_test)
	
	except KeyboardInterrupt:
		print("\033[33mStopped by user.\033[0m")
		sys.exit(1)

	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()
