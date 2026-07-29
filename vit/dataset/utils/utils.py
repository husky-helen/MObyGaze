import torch
import os
import glob
import numpy as np

from constants import MODALITY_TO_CONCEPTS


def listconcepts2onehot(list_concepts, modality="visual"):
	target_concepts = MODALITY_TO_CONCEPTS[modality]
	res = torch.zeros(len(target_concepts))
	for i, word in enumerate(target_concepts):
		if word in list_concepts:
			res[i] = 1

	return res


def tensor2maxseqlen(feat, max_seq_len):
	new_feat = torch.zeros((max_seq_len, feat.shape[1]))
	mask = torch.ones(max_seq_len)
	fs = feat.shape[0]
	if fs < max_seq_len:  # zero padding if shorter than max_seq_len
		new_feat[:fs, :] = feat
		mask[fs:] = torch.zeros(max_seq_len - fs)
	else:
		r = torch.linspace(0, len(feat), max_seq_len + 1,
						   dtype=int)  # linspace so we have 160 values - https://pytorch.org/docs/stable/generated/torch.linspace.html
		for i in range(max_seq_len):
			if r[i] != r[
				i + 1]:  # When r[i] and r[i+1] are not consecutive, it results in averaging multiple input features.
				new_feat[i, :] = torch.mean(feat[r[i]:r[i + 1], :], dim=0)
			else:  ##When r[i] and r[i+1] are consecutive integers, it results in a "direct copy" (though technically it's still an average, but of a single value).
				new_feat[i, :] = feat[r[i], :]
	# print(mask)
	# mask = mask @ mask.T
	return new_feat, mask.to(dtype=torch.bool)


def load_i3d(movie_folder):
	suffixes = ["flow.npy", "rgb.npy", "timestamps_ms.npy"]
	print("MOVIE FOLDER = ", movie_folder)
	# Find files that match the suffixes
	files = []
	for suffix in suffixes:
		matching_files = glob.glob(os.path.join(movie_folder, f"*{suffix}"))
		if matching_files:
			files.extend(matching_files)
	print("Files : ", files)

	# RGB
	matching_files = glob.glob(os.path.join(movie_folder, f"*rgb.npy"))
	if matching_files:
		rgb = np.load(matching_files[0])

	# FLOW
	matching_files = glob.glob(os.path.join(movie_folder, f"*flow.npy"))
	if matching_files:
		flow = np.load(matching_files[0])

	# TIMESTAMP in milliseconde
	matching_files = glob.glob(os.path.join(movie_folder, f"*timestamps_ms.npy"))
	if matching_files:
		timestamp = np.load(matching_files[0]) * 0.001  # Load + converstion from ms to s

	return rgb, flow, timestamp


def audiotensor2maxseqlen(feat, max_seq_len):
	"""
	Adjusts features to a maximum sequence length by truncating or padding.
	Returns the adjusted feature tensor and a corresponding mask.
	"""

	new_feat = torch.zeros((max_seq_len, feat.shape[-1]))  # match the feature dimension
	mask = torch.zeros(max_seq_len)  # intialize the mask with zeros (padded)

	fs = feat.shape[0]  # number of features in the input
	# print(f"Number of features (sequence length): {fs}, Max sequence length: {max_seq_len}")

	if fs <= max_seq_len:  # zero-padding if shorter than max_seq_len
		new_feat[:fs, :] = feat
		mask[:fs] = 1  # mark valid positions in the mask
	# print(f"Padding applied. Mask: {mask}")
	else:
		r = torch.linspace(0, len(feat), max_seq_len + 1, dtype=int)
		for i in range(max_seq_len):
			if r[i] != r[i + 1]:
				new_feat[i, :] = torch.mean(feat[r[i]:r[i + 1], :], dim=0)
			else:
				new_feat[i, :] = feat[r[i], :]
		mask[:] = 1  # All positions are valid after truncation
	# print(f"Truncation applied. Mask: {mask}")

	return new_feat, mask.to(dtype=torch.bool)
