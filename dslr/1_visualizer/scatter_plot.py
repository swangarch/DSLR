#!/usr/bin/python3

import pandas as pd
import sys
from load_csv import load
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def visualize_scatter(data: pd.DataFrame, colors:list, color_map:map, feature1:str, feature2:str) -> None:
	"""Visualize data."""
	
	try:
		plt.scatter(data[feature1].values, data[feature2].values, s=1, c=colors)
		title = f"Scatter:{feature1} * {feature2}"
		plt.title(title)
		plt.xlabel(feature1)
		plt.ylabel(feature2)
		legend_elements = [Patch(facecolor=color_map[house], label=house) for house in color_map]
		plt.legend(loc="upper left", handles=legend_elements)
		plt.show()
	except Exception as e:
		print("Error in scatter visualizer:", e)


def plot_scatter(data: pd.DataFrame, max_num: int, types_color:list, color_map:map) -> None:
	"""Show all the pair plots."""

	features = data.iloc[0].index

	count = 0
	for f1 in features:
		for f2 in features:
			if f1 != f2:
				count += 1
				print(f"Scatter {count}: {f1} * {f2}")
				visualize_scatter(data, types_color, color_map, f1, f2)
			if max_num is not None and count >= max_num:
				print("\033[33mDone\033[0m")
				return
	print("\033[33mDone\033[0m")
	plt.show()


def main():
	"""Program to visualize data."""

	try:
		print("\033[33mUsage: python3 scatter_plot.py <path_csv> <optional_max_plot_num>\033[0m")
		argv = sys.argv
		assert len(argv) == 2 or len(argv) == 3, "Wrong argument number."
		max_num = None
		if len(argv) == 3:
			max_num = int(argv[2])
		assert max_num is None or max_num >= 0, "Wrong max show number"
		df = load(argv[1])
		if df is None:
			sys.exit(1)
		
		types = df["Hogwarts House"].values
		color_map = {'Ravenclaw':'#1f77b4', 'Hufflepuff':'#ff7f0e', 'Gryffindor': '#d62728', 'Slytherin':'#2ca02c'}
		types_color = [color_map[house] for house in types]
		df_num = df.select_dtypes(include="number")
		plot_scatter(df_num, max_num, types_color, color_map)

	except KeyboardInterrupt:
		print("\033[33mStopped by user.\033[0m")
		sys.exit(1)

	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()

