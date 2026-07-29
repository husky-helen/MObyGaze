import numpy as np
import torch
import torch.nn.functional as F


def smooth(arr, lamda1):
	arr2 = torch.zeros_like(arr)
	arr2[:-1] = arr[1:]
	arr2[-1] = arr[-1]
	loss = torch.sum((arr2 - arr) ** 2)
	return lamda1 * loss


def sparsity(arr, lamda2):
	loss = torch.sum(arr)
	return lamda2 * loss


def ranking(scores, mask, batch_size, nb_segment, device):
	loss = torch.tensor(0., requires_grad=True)

	for i in range(batch_size): # we go through the bags of the batch
		normal_start = int(i * nb_segment)
		normal_end = int((i + 1) * nb_segment)
		normal_mask = mask[normal_start:normal_end]
		normal_indices = torch.arange(normal_start, normal_end).to(device)[normal_mask]
		anormal_start = int(i * nb_segment + batch_size * nb_segment)
		anormal_end = int((i + 1) * nb_segment + batch_size * nb_segment)
		anormal_mask = mask[anormal_start:anormal_end]
		anormal_indices = torch.arange(anormal_start, anormal_end).to(device)[anormal_mask]

		maxn = torch.max(scores[normal_indices]) # normal bag i
		maxa = torch.max(scores[anormal_indices]) # anomaly bag i
		tmp = F.relu(1. - maxa + maxn)
		loss = loss + tmp
		loss = loss + smooth(scores[anormal_indices], 8e-5)
		loss = loss + sparsity(scores[anormal_indices], 8e-5)
	return loss / batch_size
