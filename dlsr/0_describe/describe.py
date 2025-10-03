import pandas as pd
import sys
from load_csv import load


def ft_describe_one(series: pd.Series, title: str, all: dict) -> None:
	"""Function to describe the data series."""

	sum = 0.0
	count = 0
	mean = 0.0
	sortedLi = sorted(series)
	err = 0.0
	cleanLi = []
	for elem in sortedLi: 
		if elem == elem:
			sum += elem
			cleanLi.append(elem)

	count = len(cleanLi)
	mean = sum / count
	for num in cleanLi:
		err += (num - mean) ** 2
	std = (err / count) ** 0.5
	min_num = cleanLi[0]
	max_num = cleanLi[-1]
	percent25 = cleanLi[int((count) / 4)]
	percent50 = cleanLi[int((count) / 2)]
	percent75 = cleanLi[int((count) / 4 * 3)]

	all[title] = [count, mean, std, min_num, percent25, percent50, percent75, max_num]


def ft_describe(series: pd.Series)-> None:
	"""Describe all features"""

	all = {"Feature": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "max"]}

	print("\n--------------------my describe----------------------")
	for col in series:
		ft_describe_one(series[col], col, all)
	df_describ = pd.DataFrame(all)
	df_describ = df_describ.set_index("Feature")
	df_describ.index.name = None
	print(df_describ)


def additional_info(series: pd.Series):
	"""Show some additional info."""

	print("\n--------------------additional describe----------------------")
	for col in series:
		s = set()
		
		for elem in series[col]:
			if elem is not None:
				s.add(elem)

		show_set = str(list(s)) if len(list(s)) <= 5 else (str(list(s)[:5]) + "...")

		print(f"\033[33m<FEATURE--{col}>:\033[0m {show_set} \033[33m<feature type => {len(s)}>\033[0m")


def main():
	"""Test of reading data and print it."""

	try:
		print("\033[33mUsage: python3 describe.py <path_csv>\033[0m")
		argv = sys.argv
		assert len(argv) == 2, "Wrong argument number."

		pd.set_option('display.float_format', '{:.6f}'.format)
		df = load(argv[1])
		if df is None:
			sys.exit(1)
		print(df)
		
		print("\n--------------------pd describe----------------------")
		print("\033[33m", df.describe(), "\033[0m")  ##remove this, forbidden!!!!!!!!!!!!!!!!

		additional_info(df)
 
		df_num = df.select_dtypes(include="number")
		ft_describe(df_num)

		rows_with_nan = df[df_num.isna().any(axis=1)]
		print("\nLine include Nan:")
		print(rows_with_nan)

	except KeyboardInterrupt:
		print("\033[33mStopped by user.\033[0m")
		sys.exit(1)
	
	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	main()