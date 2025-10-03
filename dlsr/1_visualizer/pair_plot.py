#!/usr/bin/python3

import pandas as pd
import sys
from load_csv import load
import matplotlib.pyplot as plt


def visualize_hist(data: pd.DataFrame, feature:str, length:int, pos: int, subdfs: map, color_map:map) -> None:
	"""Visualize data."""
	
	try:
		plt.subplot(length, length, pos)
		plt.hist(data[feature].values, bins=100, rwidth=1, color="lightgrey", label="Total")
		for house in subdfs:
			subdf = subdfs[house]
			plt.hist(subdf[feature].values, bins=100, rwidth=1, color=color_map[house], alpha=0.7, label=house)
		plt.xticks([])
		plt.yticks([])
	except Exception as e:
		print("Error in histogram visualizer:", e)


def visualize_scatter(data: pd.DataFrame, feature1:str, feature2:str, length:int, pos: int, types_color:list) -> None:
	"""Visualize pair plot data."""
	
	try:
		plt.subplot(length, length, pos)
		plt.scatter(data[feature1].values, data[feature2].values, s=0.1, c=types_color)
		plt.xticks([])
		plt.yticks([])
	except Exception as e:
		print("Error in scatter visualizer:", e)


def pair_plot(data: pd.DataFrame, types_color:list, subdfs_map:map, color_map:map) -> None:
	"""Show all the pair plots."""

	fig = plt.gcf()
	fig.suptitle("Pair plot on all features", fontsize=14)

	features = data.iloc[0].index
	num_features = len(features)
	count = 0
	for f1 in features:
		for f2 in features:
			count += 1
			
			if f1 != f2:
				visualize_scatter(data, f1, f2, num_features, count, types_color)
			else:
				visualize_hist(data, f1, num_features, count, subdfs_map, color_map)

			row = (count - 1) // num_features
			col = (count - 1) % num_features
			# put col row label
			if col == 0:
				plt.ylabel(f1, fontsize=10, rotation=0, ha="right")
			if row == num_features - 1:
				plt.xlabel(features[col], fontsize=10, ha="center")
	
	plt.show()


def main():
	"""Program to visualize data."""

	try:
		print("\033[33mUsage: python3 pair_plot.py <path_csv>\033[0m")
		argv = sys.argv
		assert len(argv) == 2, "Wrong argument number."
		df = load(argv[1])
		if df is None:
			sys.exit(1)

		houses = ['Ravenclaw', 'Hufflepuff', 'Gryffindor', 'Slytherin']
		subdfs_map = dict()
		for house in houses:
			result = df[df["Hogwarts House"] == house]
			subdfs_map[house] = result

		types = df["Hogwarts House"].values
		color_map = {'Ravenclaw':'#1f77b4', 'Hufflepuff':'#ff7f0e', 'Gryffindor': '#d62728', 'Slytherin':'#2ca02c'}
		types_color = [color_map[house] for house in types]
		df_num = df.select_dtypes(include="number")
		pair_plot(df_num, types_color, subdfs_map, color_map)

	except KeyboardInterrupt:
		print("\033[33mStopped by user.\033[0m")
		sys.exit(1)

	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()

