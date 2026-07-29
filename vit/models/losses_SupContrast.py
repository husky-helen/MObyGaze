# Inspired from https://github.com/HobbitLong/SupContrast/losses.py

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


# N.B.: in the original SupCon implementation, when calling, the shapes should be
# features: [bsz, n_views, f_dim] where `n_views` is the nb of augmentations for each observation ;  labels: [bsz]
# and features should be normalized (should be done properly in the perspective of gradient back-propagation)
# I adapted the code so that, if it's not the case, it is performed automatically during the call

# N.B. : This implementation works, but be careful to feed it with precise tensors (dtype torch.float64),
# else the computation may diverge (exp results of large negative are approximated by 0.)

class SupConLoss(nn.Module):
	"""Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
	It also supports the unsupervised contrastive loss in SimCLR"""

	def __init__(self, temperature=0.07, contrast_mode='all',
				 base_temperature=0.07):
		super(SupConLoss, self).__init__()
		self.temperature = temperature
		self.contrast_mode = contrast_mode
		self.base_temperature = base_temperature

	def forward(self, features, labels=None, mask=None):
		"""Compute loss for model. If both `labels` and `mask` are None,
		it degenerates to SimCLR unsupervised loss:
		https://arxiv.org/pdf/2002.05709.pdf

		Args:
			features: hidden vector of shape [bsz, n_views, ...].
			labels: ground truth of shape [bsz].
			mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
				has the same class as sample i. Can be asymmetric.
		Returns:
			A loss scalar.
		"""
		device = (torch.device('cuda')
				  if features.is_cuda
				  else torch.device('cpu'))

		if len(features.shape) < 2:
			raise ValueError('`features` needs to be [bsz, n_views, ...],'
							 'at least 3 dimensions are required')
		if len(features.shape) == 2:  # MY ADDITIONS
			# It means the features have shape [bsz, D] so we artificially include the first dimension n_views=1
			# to match the original code
			features = torch.unsqueeze(features, 1)
		if len(features.shape) > 3:
			features = features.view(features.shape[0], features.shape[1], -1)

		# Moreover this implementation expects normalized features, so, to avoid any bug, we implement it directly
		# (! in a way that is differentiable)
		features = F.normalize(features, dim=-1, p=2)  # MY ADDITIONS

		batch_size = features.shape[0]
		if labels is not None and mask is not None:
			raise ValueError('Cannot define both `labels` and `mask`')
		elif labels is None and mask is None:
			mask = torch.eye(batch_size, dtype=torch.float32).to(device)
		elif labels is not None:
			labels = labels.contiguous().view(-1, 1)
			if labels.shape[0] != batch_size:
				raise ValueError('Num of labels does not match num of features')
			mask = torch.eq(labels, labels.T).float().to(device)
		else:
			mask = mask.float().to(device)

		contrast_count = features.shape[1]
		contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
		if self.contrast_mode == 'one':
			anchor_feature = features[:, 0]
			anchor_count = 1
		elif self.contrast_mode == 'all':
			anchor_feature = contrast_feature
			anchor_count = contrast_count
		else:
			raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

		# compute logits
		anchor_dot_contrast = torch.div(
			torch.matmul(anchor_feature, contrast_feature.T),
			self.temperature)
		# for numerical stability
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach()

		# tile mask
		mask = mask.repeat(anchor_count, contrast_count)
		# mask-out self-contrast cases
		logits_mask = torch.scatter(
			torch.ones_like(mask),
			1,
			torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
			0
		)
		mask = mask * logits_mask

		# compute log_prob
		exp_logits = torch.exp(logits)

		exp_logits = torch.where(exp_logits < torch.finfo(exp_logits.dtype).tiny, torch.finfo(exp_logits.dtype).tiny,
								 exp_logits)
		# MY ADDITION : this prevents errors (that seem due to some exp logits being 0 (mathematically impossible),
		# but this way the pushing_away training (during which these errors occurred) diverges

		exp_logits = exp_logits * logits_mask
		log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

		# compute mean of log-likelihood over positive
		# modified to handle edge cases when there is no positive pair
		# for an anchor point.
		# Edge case e.g.:-
		# features of shape: [4,1,...]
		# labels:            [0,1,1,2]
		# loss before mean:  [nan, ..., ..., nan]
		mask_pos_pairs = mask.sum(1)
		mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
		mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

		# loss
		loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
		loss = loss.view(anchor_count, batch_size).mean()

		return loss


class IntraClusterSupConLoss(nn.Module):

	def __init__(self, contrastive_type, temperature=0.1, weighted=False):
		"""
		:param contrastive_type: the type of contrastive loss ('supcon', 'pushing_away', 'mixed')
		:param temperature: temperature parameter of the loss
		:param weighted: if the sum over the clusters is weighted (by the inverse of their size)
		"""
		super(IntraClusterSupConLoss, self).__init__()
		self.supcon_loss_fun = SupConLoss(temperature=temperature, base_temperature=temperature)
		# we prefer the original SupCon implementation, as it enables to use a mask instead of raw labels
		self.weighted = weighted
		self.contrastive_type = contrastive_type
		# Luckily for the implementation of pushing away, the original SupCon implementation enables
		# to compute the loss based on a contrastive mask instead of actual labels :
		# "mask: contrastive mask of shape [bsz, bsz],
		# mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric."

	def forward(self, outputs):
		feats = outputs['feats']  # feats shape: [B, D]
		labels = outputs['labels']  # labels shape: [B]
		ids = outputs['ids']  # cluster ids shape: [B]
		if len(labels.shape) > 1:  # typically given labels may have shape [B, 1], which is useful in some contexts but not here
			labels = labels.view(-1)

		# From the labels, we construct the contrastive mask
		# (labels can be used directly in the classical case, but we will need the mask for the inverse case ;
		# thus defining a mask will enable easy generalization of the following of the code)
		supcon_mask = torch.stack([labels == labels[i] for i in range(len(labels))])
		pushing_away_mask = torch.bitwise_not(supcon_mask)
		# N.B. : In this pushing_away_mask, an anchor i does not even belong to its own class,
		# but it does not affect the result, as anyway i does not belong to A(i), P(i) or N(i)

		device = (torch.device('cuda')
				  if feats.is_cuda
				  else torch.device('cpu'))

		unique_ids, id_counts = torch.unique(ids, return_counts=True)
		relevant_ids, relevant_counts = unique_ids[id_counts > 1], id_counts[id_counts > 1]
		# the SupCon loss requires the A(i) set of each anchor i to be non-empty ;
		# in other words each cluster should contain at least 2 elements

		if len(relevant_ids) == 0:
			return torch.Tensor([0.]).to(device)
		else:
			supcon_loss = torch.Tensor([0.]).to(device)
			pushing_away_loss = torch.Tensor([0.]).to(device)
			# (somehow using torch.sum over the tensor of cluster losses does not back-propagate... so, ugly loop)
			for k, n_k in zip(relevant_ids, relevant_counts):
				weight = (1 / n_k) if self.weighted else 1.
				supcon_loss += weight * self.supcon_loss_fun(features=feats[ids == k],
															 mask=supcon_mask[ids == k][:, ids == k])
				pushing_away_loss += weight * self.supcon_loss_fun(features=feats[ids == k],
																   mask=pushing_away_mask[ids == k][:, ids == k])
			pushing_away_loss = - pushing_away_loss

			scaling_weight = (100 / len(relevant_ids)) if self.weighted else (100 / labels.shape[0])
			supcon_loss *= scaling_weight
			pushing_away_loss *= scaling_weight
			# The point of scaling is to make the loss values comparable between different batch sizes
			# Theoretically, scaling the loss should not impact Adam optimizer
			# The 100 factor is just there so that values remain readable

			# print(f"supcon_loss : {supcon_loss.item()} ; pushing_away_loss : {pushing_away_loss.item()}")

			if self.contrastive_type == "supcon":
				returned_loss = supcon_loss
			elif self.contrastive_type == "pushing_away":
				returned_loss = pushing_away_loss
			elif self.contrastive_type == "mixed":
				# N.B. : one may want to introduce a weighting coefficient
				returned_loss = supcon_loss + pushing_away_loss
			else:
				raise Exception(f"Unknown contrastive type '{self.contrastive_type}'")

			return {'returned_loss': returned_loss, 'supcon_loss': supcon_loss, 'pushing_away_loss': pushing_away_loss}


class GlobalSupConLoss(nn.Module):

	def __init__(self, contrastive_type, temperature=0.1):
		"""
		:param contrastive_type: the type of contrastive loss ('supcon', 'pushing_away', 'mixed')
		:param temperature: temperature parameter of the loss
		"""
		super(GlobalSupConLoss, self).__init__()
		self.supcon_loss_fun = SupConLoss(temperature=temperature, base_temperature=temperature)
		# we prefer the original SupCon implementation, as it enables to use a mask instead of raw labels
		self.contrastive_type = contrastive_type
		# Luckily for the implementation of pushing away, the original SupCon implementation enables
		# to compute the loss based on a contrastive mask instead of actual labels :
		# "mask: contrastive mask of shape [bsz, bsz],
		# mask_{i,j}=1 if sample j has the same class as sample i. Can be asymmetric."

	def forward(self, outputs):
		feats = outputs['feats']  # feats shape: [B, D]
		labels = outputs['labels']  # labels shape: [B]
		# No need of the ids here
		if len(labels.shape) > 1:  # typically given labels may have shape [B, 1], which is useful in some contexts but not here
			labels = labels.view(-1)

		# From the labels, we construct the contrastive mask
		# (labels can be used directly in the classical case, but we will need the mask for the inverse case ;
		# thus defining a mask will enable easy generalization of the following of the code)
		supcon_mask = torch.stack([labels == labels[i] for i in range(len(labels))])
		pushing_away_mask = torch.bitwise_not(supcon_mask)
		# N.B. : In this pushing_away_mask, an anchor i does not even belong to its own class,
		# but it does not affect the result, as anyway i does not belong to A(i), P(i) or N(i)

		supcon_loss = self.supcon_loss_fun(features=feats, mask=supcon_mask) * (100 / labels.shape[0])
		pushing_away_loss = - self.supcon_loss_fun(features=feats, mask=pushing_away_mask) * (100 / labels.shape[0])
		# The point of scaling is to make the loss values comparable between different batch sizes
		# Theoretically, scaling the loss should not impact Adam optimizer
		# The 100 factor is just there so that values remain readable

		# print(f"supcon_loss : {supcon_loss.item()} ; pushing_away_loss : {pushing_away_loss.item()}")

		if self.contrastive_type == "supcon":
			returned_loss = supcon_loss
		elif self.contrastive_type == "pushing_away":
			returned_loss = pushing_away_loss
		elif self.contrastive_type == "mixed":
			# N.B. : one may want to introduce a weighting coefficient
			returned_loss = supcon_loss + pushing_away_loss
		else:
			raise Exception(f"Unknown contrastive type '{self.contrastive_type}'")

		# We return all losses for monitoring
		return {'returned_loss': returned_loss, 'supcon_loss': supcon_loss, 'pushing_away_loss': pushing_away_loss}
