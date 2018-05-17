import numpy as np


def binary_to_categorical(binary):
	"""
	Function to convert binary labels into categorical.
	"""
	cat = np.zeros(binary.shape[0])
	for i in range(binary.shape[0]):
		cat[i] = binary[i,:].argmax()
	return cat
