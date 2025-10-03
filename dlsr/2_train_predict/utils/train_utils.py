#!/usr/bin/python3

import numpy as np
from numpy import ndarray as array
from pandas import DataFrame as dataframe


def normalize(li: array, range: tuple) -> array:
	"""Normalize list value, mapped into range of 0:1"""

	min_val, max_val = range
	return (li - min_val) / (max_val - min_val)


def cross_entropy_loss(predict:array, truth:array) -> None:
	"""Calculate the loss"""

	pass

def sigmoid(arr:array) -> array:
	"""Apply sigmoid to data"""

	return 1.0 / (1 + np.e ** (-arr))


def prob_predict(weight:array, fnorm:list, length: int) -> array:
	"""Calculate the training prediction"""

	res = np.zeros(length, dtype=np.float32)
	for i, f in enumerate(fnorm):
		res += weight[i + 1] * f
	res += weight[0]
	return sigmoid(res)


def preproc_onevsall(arr:array, catName: str) -> array:
	"""preprocess class one vs all"""

	return np.array([1 if h == catName else 0 for h in arr])


def cal_grad(prediction: array, groundTruth: array, features: list, weight:array) -> array:
	"""Calculate the gradient of loss to weight"""

	diff = prediction - groundTruth
	length = len(weight)
	gradient = np.zeros(length, dtype=np.float32)
	for i in range(length):
		if i == 0: #bias
			gradient[i] = (np.nansum(diff) / length)
		else: #weights
			gradient[i] = (np.nansum(diff * features[i - 1]) / length)

	return gradient


def count_correct(title:str, prediction:array, truth: array) -> str:
	"""Count prediction truth"""

	count = 0
	length = len(truth)
	for i, num in enumerate(truth):
		if (num == prediction[i]):
			count += 1

	return f"\033[33m{title} <CORRECT> {count}/{length}   <RATE> {count / float(length) * 100:2f}%\033[0m"


def clean_data(df: dataframe, method:str="dropnan") -> dataframe:
	"""Clean data, drop nan value line"""

	if method == "dropnan":
		num_cols = df.select_dtypes(include=["number"]).columns
		df.dropna(subset=num_cols)
	elif method == "raw":
		return df
	elif method == "mean":
		return df.fillna(df.mean(numeric_only=True))
	elif method == "median":
		return df.fillna(df.median(numeric_only=True))
	return df


def split_data(df: dataframe, frac=0.8, seed=412) -> tuple:
	"""Split the data into test data and validation data"""

	train_df = df.sample(frac=frac, random_state=seed)
	test_df = df.drop(train_df.index)
	return train_df, test_df
