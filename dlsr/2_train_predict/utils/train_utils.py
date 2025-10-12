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

	p = predict.flatten()
	eps = 1e-12
	p = np.clip(p, eps, 1-eps)
	loss = -np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p))
	return loss


def sigmoid(arr:array) -> array:
	"""Apply sigmoid to data"""

	return 1.0 / (1 + np.e ** (-arr))


def prob_predict(weight:array, fnorm:array) -> array:
	"""Calculate the training prediction"""

	return sigmoid(weight @ fnorm)


def preproc_onevsall(arr:array, catName: str) -> array:
	"""preprocess class one vs all"""

	return np.array([1 if h == catName else 0 for h in arr])


def cal_grad(prediction: array, groundTruth: array, features: array, weight:array) -> array:
	"""Calculate the gradient of loss to weight"""

	length = len(weight)
	diff = (prediction - groundTruth) / length
	gradient = diff @ features.T
	return gradient


def count_correct(title:str, prediction:array, truth: array, has_loss: bool) -> str:
	"""Count prediction truth"""

	count = 0
	t = truth.flatten()
	length = len(t)
	p = prediction.flatten()
	for i, num in enumerate(t):
		if (num == p[i]):
			count += 1
	if has_loss:
		loss = cross_entropy_loss(prediction, truth)
		return f"\033[33m{title} <Correct> {count}/{length}   <Accuracy> {count / float(length) * 100:2f}%   <Loss> {loss:.4f}\033[0m"
	else:
		return f"\033[33m{title} <Correct> {count}/{length}   <Accuracy> {count / float(length) * 100:2f}%\033[0m"


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


def save_weights(weight: array, path:str) -> None:
    """Save the weights."""

    try:
        np.savetxt(path, weight, delimiter=",", fmt="%f")
    
    except Exception as e:
        print("Error:", e)