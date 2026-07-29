import torch
from torch import nn
from torch.autograd import Function

from models.vit_models import RepresentationViT, ClassificationHead, RepresentationDifferentialViT


# First we define the GradientReversal layer that is at the core of DANN
# From https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/5
# (If this does not work, see this other proposal: https://github.com/tadeephuy/GradientReversal/tree/master)

class GradReverse(Function):
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output.neg()


def grad_reverse(x):
	return GradReverse.apply(x)
# We can use it in the forward call as
# x = grad_reverse(x)


class DANNViT(nn.Module):
	def __init__(
			self,
			*,
			# REPRESENTATION MODEL
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			# CLASSIFICATION HEADS
			num_classes,
			num_domains,
			classif_dropout=0.,
			# OTHER
			return_representation=False
	):
		super().__init__()

		self.representation_model = RepresentationViT(feature_dim=feature_dim, max_seq_length=max_seq_length,
													  dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
													  dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)
		self.classification_head = ClassificationHead(dim=dim, dropout=classif_dropout, num_classes=num_classes)
		self.domain_classifier = ClassificationHead(dim=dim, dropout=classif_dropout, num_classes=num_domains)

		self.return_representation = return_representation

	def forward(self, x, mask):
		embedding = self.representation_model(x, mask)
		classif_pred = self.classification_head(embedding)
		domain_pred = self.domain_classifier(grad_reverse(embedding))

		if self.return_representation:
			return embedding, classif_pred, domain_pred
		else:
			return classif_pred, domain_pred


class DifferentialDANNViT(nn.Module):
	def __init__(
			self,
			*,
			# REPRESENTATION MODEL
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			# CLASSIFICATION HEADS
			num_classes,
			num_domains,
			classif_dropout=0.,
			# OTHER
			return_representation=False
	):
		super().__init__()
		self.representation_model = RepresentationDifferentialViT(feature_dim=feature_dim, max_seq_length=max_seq_length,
													  dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool,
													  dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)
		self.classification_head = ClassificationHead(dim=dim, dropout=classif_dropout, num_classes=num_classes)
		self.domain_classifier = ClassificationHead(dim=dim, dropout=classif_dropout, num_classes=num_domains)

		self.return_representation = return_representation

	def forward(self, x, mask, ids):
		embedding = self.representation_model(x, mask, ids)
		classif_pred = self.classification_head(embedding)
		domain_pred = self.domain_classifier(grad_reverse(embedding))

		if self.return_representation:
			return embedding, classif_pred, domain_pred
		else:
			return classif_pred, domain_pred
