#!/usr/bin/python3

import pandas as pd
import sys
from load_csv import load
import matplotlib.pyplot as plt


def visualize_hist(data: pd.DataFrame, feature:str, subdfs: map, color_map:map) -> None:
	"""Visualize data."""
	
	try:
		plt.hist(data[feature].values, bins=100, rwidth=1, color="lightgrey", label="Total")
		for house in subdfs:
			subdf = subdfs[house]
			plt.hist(subdf[feature].values, bins=100, rwidth=1, color=color_map[house], alpha=0.7, label=house)
		title = f"histogram:{feature}"
		plt.title(title)
		plt.xlabel(feature)
		plt.ylabel("Value")
		plt.legend(loc="upper left")
		plt.show()
	except Exception as e:
		print("Error in scatter visualizer:", e)


def plot_hist(data: pd.DataFrame, subdfs:map, color_map:map) -> None:
	"""Show all the pair plots."""

	features = data.iloc[0].index

	count = 0
	for f1 in features:
		count += 1
		print(f"Histogram {count}: {f1}")
		visualize_hist(data, f1, subdfs, color_map)
	
	print("\033[33mDone.\033[0m")
	plt.show()


def main():
	"""Program to visualize data."""

	try:
		print("\033[33mUsage: python3 histogram.py <path_csv>\033[0m")
		argv = sys.argv
		assert len(argv) == 2 or len(argv ), "Wrong argument number."
		df = load(argv[1])
		if df is None:
			sys.exit(1)

		houses = ['Ravenclaw', 'Hufflepuff', 'Gryffindor', 'Slytherin']
		subdfs_map = dict()
		for house in houses:
			result = df[df["Hogwarts House"] == house]
			subdfs_map[house] = result

		color_map = {'Ravenclaw':'#1f77b4', 'Hufflepuff':'#ff7f0e', 'Gryffindor': '#d62728', 'Slytherin':'#2ca02c'}
		df_num = df.select_dtypes(include="number")
		plot_hist(df_num, subdfs_map, color_map)

	except KeyboardInterrupt:
		print("\033[33mStopped by user.\033[0m")
		sys.exit(1)

	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()
