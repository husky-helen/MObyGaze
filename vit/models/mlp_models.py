import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from copy import deepcopy


class RepresentationMLP(nn.Module):
	def __init__(self, feature_dim, dim):
		super().__init__()
		self.fc1 = nn.Linear(feature_dim, dim)
		self.bn1 = nn.BatchNorm1d(dim)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.relu(x)
		return x


class BigRepresentationMLP(nn.Module):
	def __init__(self, feature_dim, dim, dropout=0.1):
		super().__init__()
		self.big_representation_mlp = nn.Sequential(  # FSL neurips experiment layer
			nn.Linear(feature_dim, feature_dim // 2),
			nn.BatchNorm1d(feature_dim // 2),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(feature_dim // 2, dim),
			nn.BatchNorm1d(dim),
			nn.ReLU(),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		return self.big_representation_mlp(x)


class MLP(nn.Module):
	def __init__(self, feature_dim, dim, num_classes, representation_architecture, dropout=0., return_representation=False,
			detach_representation=True
			# usually, as I only return the representation as a debugging/analysis feature, I can detach it
			# In other cases however (only CEM, for now), this representation is used in the computation,
			# thus gradient descent, and should not be detached
	):
		super().__init__()
		big_mlp = "bigmlp" in representation_architecture
		self.representation_mlp = BigRepresentationMLP(feature_dim=feature_dim, dim=dim) if big_mlp else (
			RepresentationMLP(feature_dim=feature_dim, dim=dim))

		self.mlp_head = nn.Sequential(  # FSL neurips experiment layer
			nn.Linear(dim, dim // 2),
			nn.BatchNorm1d(dim // 2),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(dim // 2, num_classes)
		)
		# This way, our MLP model has a similar classification head to the ViT models ;
		# and its representation is a single ReLU layer

		# The other choice was
		#self.mlp_head = nn.Sequential(
		#	nn.Dropout(dropout),
		#	nn.Linear(dim, num_classes)
		#)
		# so that the full MLP is equivalent to the ViT classification head

		self.to_latent = nn.Identity()

		self.return_representation = return_representation
		self.detach_representation = detach_representation

	def forward(self, x, mask=None):

		x = self.representation_mlp(x)
		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()
		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class PreConceptMLP(nn.Module):
	def __init__(self, feature_dim, dim, num_classes, dropout=0., return_representation=False,
			detach_representation=True
			# usually, as I only return the representation as a debugging/analysis feature, I can detach it
			# In other cases however (only CEM, for now), this representation is used in the computation,
			# thus gradient descent, and should not be detached
	):
		super().__init__()
		self.representation_mlp = nn.Sequential(
			nn.Linear(in_features=feature_dim, out_features=dim, bias=True),
			nn.LeakyReLU(negative_slope=0.01),
			nn.Linear(in_features=dim, out_features=dim, bias=True),
			nn.LeakyReLU(negative_slope=0.01),
			nn.Linear(in_features=dim, out_features=dim, bias=True)
		)

		self.mlp_head = nn.Sequential(  # FSL neurips experiment layer
			nn.Linear(dim, dim // 2),
			nn.BatchNorm1d(dim // 2),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(dim // 2, num_classes)
		)

		self.to_latent = nn.Identity()

		self.return_representation = return_representation
		self.detach_representation = detach_representation

	def forward(self, x, mask=None):

		x = self.representation_mlp(x)
		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()
		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


# The "MLP" model and "ClassificationHead" are kinda similar ;
# however, while MLP will be used in the code as a full model (from raw embeddings to output classification),
# ClassificationHead is only designed as a classification head on top of a pre-computed representation ;
# in particular, ClassificationHead does not return any representation
# (Also, according to the remark in MLP code, the architecture is not completely identical)


class DifferentialMLP(MLP):
	def __init__(self, feature_dim, dim, num_classes, representation_architecture, dropout=0., return_representation=False,
			detach_representation=True):
		super().__init__(feature_dim=feature_dim, dim=dim, num_classes=num_classes, dropout=dropout,
						 return_representation=return_representation, representation_architecture=representation_architecture,
						 detach_representation=detach_representation)

	def forward(self, x, ids):

		x = self.representation_mlp(x)
		output_x = x.clone()  # we will modify the rows from this tensor

		# Here we do the differential operation
		for id in torch.unique(ids):
			# We compute the representation of the context
			id_context = torch.mean(x[ids == id], dim=0)
			# We compute x representation relatively to this context
			output_x[ids == id] -= id_context  # see ComplexDifferentialViT to combine x and id_context in a more elaborate way

		if self.detach_representation:
			repr = output_x.detach().clone()
		else:
			repr = output_x.clone()

		output_x = self.to_latent(output_x)

		if self.return_representation:
			return repr, self.mlp_head(output_x)
		else:
			return self.mlp_head(output_x)


class GatedDifferentialMLP(MLP):
	def __init__(self, feature_dim, dim, num_classes, representation_architecture, dropout=0., return_representation=False,
			detach_representation=True):
		super().__init__(feature_dim=feature_dim, dim=dim, num_classes=num_classes, dropout=dropout,
						 return_representation=return_representation, representation_architecture=representation_architecture,
						 detach_representation=detach_representation)

		self.gating_vit = torch.nn.Linear(dim, 1)

	def forward(self, x, ids):

		x = self.representation_mlp(x)
		x_gating = torch.sigmoid(self.gating_vit(x))

		# Here we do the differential operation
		for id in torch.unique(ids):
			# We compute the representation of the context
			id_context = torch.mean(x[ids == id], dim=0)
			gated_id_context = x_gating[ids == id] * id_context
			# We compute x representation relatively to this context
			x = x.clone() # this is necessary to avoid autograd error because of in-place operation
			# (as x[...] = x[...] - y is an in-place operation
			x[ids == id] = x[ids == id] - gated_id_context

		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()

		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class ComplexGatedDifferentialMLP(MLP):
	def __init__(self, feature_dim, dim, num_classes, representation_architecture, dropout=0., return_representation=False,
			detach_representation=True):
		super().__init__(feature_dim=feature_dim, dim=dim, num_classes=num_classes, dropout=dropout,
						 return_representation=return_representation, representation_architecture=representation_architecture,
						 detach_representation=detach_representation)

		self.gating_vit = torch.nn.Linear(2*dim, 1)

	def forward(self, x, ids):

		x = self.representation_mlp(x)

		id_contexts = dict()
		# Here we do the differential operation
		for id in torch.unique(ids):
			# We compute the representation of the context
			id_context = torch.mean(x[ids == id], dim=0)
			# We compute x representation relatively to this context
			id_contexts[int(id)] = id_context

		corresponding_contexts = torch.stack([id_contexts[int(id_x)] for id_x in ids])
		extended_x = torch.cat([x, corresponding_contexts], dim=1)
		x_gating = torch.sigmoid(self.gating_vit(extended_x))

		x = x - x_gating * corresponding_contexts

		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()

		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class AttentionLikeGatedDifferentialMLP(MLP):
	def __init__(self, feature_dim, dim, num_classes, representation_architecture, dropout=0., return_representation=False,
			detach_representation=True):
		super().__init__(feature_dim=feature_dim, dim=dim, num_classes=num_classes, dropout=dropout,
						 return_representation=return_representation, representation_architecture=representation_architecture,
						 detach_representation=detach_representation)

		self.mlp_1 = torch.nn.Linear(dim, dim // 2)
		self.mlp_2 = torch.nn.Linear(dim, dim // 2)

	def forward(self, x, ids):

		x = self.representation_mlp(x)

		id_contexts = dict()
		# Here we do the differential operation
		for id in torch.unique(ids):
			# We compute the representation of the context
			id_context = torch.mean(x[ids == id], dim=0)
			# We compute x representation relatively to this context
			id_contexts[int(id)] = id_context

		corresponding_contexts = torch.stack([id_contexts[int(id_x)] for id_x in ids])

		x_gating =  torch.sigmoid((self.mlp_1(x) @ self.mlp_2(corresponding_contexts).T).diag())
		# (self.mlp_1(x) @ self.mlp_2(corresponding_contexts).T).diag() is nothing else than batch-wise dot-product
		x_gating = x_gating.unsqueeze(1)
		# We still introduce sigmoid to bound values in [0,1]

		x = x - x_gating * corresponding_contexts

		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()

		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class APosterioriDifferentialMLP(nn.Module):
	def __init__(
			self,
			*,
			backbone,
			dim, # the dimension of the representation returned by the backbone,
			dropout,
			num_classes,
			return_representation=False,
			detach_representation=True
	):
		super().__init__()

		self.backbone = backbone
		# We freeze the weights of the backbone
		for param in self.backbone.parameters():
			param.requires_grad = False

		self.mlp_head = nn.Sequential(  # FSL neurips experiment layer
			nn.Linear(dim, dim // 2),
			nn.BatchNorm1d(dim // 2),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(dim // 2, num_classes)
		)

		self.to_latent = nn.Identity()

		self.return_representation = return_representation
		self.detach_representation = detach_representation

	def forward(self, x, ids):

		x, logits = self.backbone(x) # we must ensure that return_representation=True for the backbone
		output_x = x.clone()  # we will modify the rows from this tensor

		# Here we do the differential operation
		for id in torch.unique(ids):
			# We compute the representation of the context
			id_context = torch.mean(x[ids == id], dim=0)
			# We compute x representation relatively to this context
			output_x[ids == id] -= id_context

		if self.detach_representation:
			repr = output_x.detach().clone()
		else:
			repr = output_x.clone()

		output_x = self.to_latent(output_x)

		if self.return_representation:
			return repr, self.mlp_head(output_x)
		else:
			return self.mlp_head(output_x)


class ComplexDifferentialMLP(MLP):
	def __init__(self, feature_dim, dim, num_classes, representation_architecture, dropout=0., return_representation=False,
			detach_representation=True):
		super().__init__(feature_dim=feature_dim, dim=dim, num_classes=num_classes, dropout=dropout,
						 return_representation=return_representation, representation_architecture=representation_architecture,
						 detach_representation=detach_representation)

		self.complex_mlp = nn.Sequential(
			nn.Linear(2*dim, dim),
			nn.BatchNorm1d(dim),
			nn.ReLU(),
			nn.Dropout(dropout)
		)

	def forward(self, x, ids):

		x = self.representation_mlp(x)

		id_contexts = dict()
		# Here we do the differential operation
		for id in torch.unique(ids):
			# We compute the representation of the context
			id_context = torch.mean(x[ids == id], dim=0)
			# We compute x representation relatively to this context
			id_contexts[int(id)] = id_context

		corresponding_contexts = torch.stack([id_contexts[int(id_x)] for id_x in ids])
		x = torch.cat([x, corresponding_contexts], dim=1)
		x = self.complex_mlp(x)

		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()

		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


# The class below could have somehow been implemented inside DifferentialVit, but this way we will be able to easily delete it
class DifferentialMLPNegative(MLP):
	def __init__(self, feature_dim, dim, num_classes, representation_architecture, dropout=0., return_representation=False,
			detach_representation=True):
		super().__init__(feature_dim=feature_dim, dim=dim, num_classes=num_classes, dropout=dropout,
						 return_representation=return_representation, representation_architecture=representation_architecture,
						 detach_representation=detach_representation)

	def forward(self, x, ids, binary_targets):

		x = self.representation_mlp(x)
		output_x = x.clone()  # we will modify the rows from this tensor

		# Here we do the differential operation
		if binary_targets is not None:
			for id in torch.unique(ids):
				# We compute the representation of the context
				context_booleans = torch.bitwise_and(ids == id, binary_targets == 0)
				id_context = torch.mean(x[context_booleans], dim=0) if torch.any(context_booleans) else 0.
				# We compute x representation relatively to this context
				output_x[ids == id] -= id_context # in the future, we ay want to combine x and id_context in a more elaborate way
		else:
			# In test and validation phase, if no cheating,
			# we cannot access real targets so we just compute the mean on the movie
			for id in torch.unique(ids):
				id_context = torch.mean(x[ids == id], dim=0)
				output_x[ids == id] -= id_context

		if self.detach_representation:
			repr = output_x.detach().clone()
		else:
			repr = output_x.clone()

		output_x = self.to_latent(output_x)

		if self.return_representation:
			return repr, self.mlp_head(output_x)
		else:
			return self.mlp_head(output_x)


class DifferentialMLPSelective(MLP):
	def __init__(self, feature_dim, dim, num_classes, representation_architecture, dropout=0., return_representation=False,
			detach_representation=True):
		super().__init__(feature_dim=feature_dim, dim=dim, num_classes=num_classes, dropout=dropout,
						 return_representation=return_representation, representation_architecture=representation_architecture,
						 detach_representation=detach_representation)

	def forward(self, x, ids):

		k = 8 # number of closest neighbors we consider

		x = self.representation_mlp(x)
		output_x = x.clone()  # we will modify the rows from this tensor

		# Here we do the differential operation
		# N.B. : This may not be the most optimal, but it is correct
		normalized_x = F.normalize(x, dim=-1, p=2)  # [N, D]
		similarity = normalized_x @ normalized_x.T
		for i in range(len(x)):
			id = ids[i]
			# First we select the x rows of the appropriate id and label
			context_booleans = ids == id
			context_x = x[context_booleans]
			# Then we consider similarity on those elements
			context_similarity = similarity[i, context_booleans]  # as many elements as context_x

			target_neighbors = torch.argsort(context_similarity, descending=True)[:k]
			context = torch.mean(context_x[target_neighbors], dim=0) if len(target_neighbors) > 0 else 0.
			output_x[i] -= context

		if self.detach_representation:
			repr = output_x.detach().clone()
		else:
			repr = output_x.clone()

		output_x = self.to_latent(output_x)

		if self.return_representation:
			return repr, self.mlp_head(output_x)
		else:
			return self.mlp_head(output_x)


class DifferentialMLPSelectiveNegative(MLP):
	def __init__(self, feature_dim, dim, num_classes, representation_architecture, dropout=0., return_representation=False,
			detach_representation=True):
		super().__init__(feature_dim=feature_dim, dim=dim, num_classes=num_classes, dropout=dropout,
						 return_representation=return_representation, representation_architecture=representation_architecture,
						 detach_representation=detach_representation)

	def forward(self, x, ids, binary_targets):

		k = 8 # number of closest neighbors we consider

		x = self.representation_mlp(x)
		output_x = x.clone()  # we will modify the rows from this tensor

		# Here we do the differential operation
		# N.B. : This may not be the most optimal, but it is correct
		if binary_targets is not None:
			normalized_x = F.normalize(x, dim=-1, p=2)  # [N, D]
			similarity = normalized_x @ normalized_x.T
			for i in range(len(x)):
				id = ids[i]
				# First we select the x rows of the appropriate id and label
				context_booleans = torch.bitwise_and(ids == id, binary_targets == 0)
				context_x = x[context_booleans]
				# Then we consider similarity on those elements
				context_similarity = similarity[i, context_booleans] # as many elements as context_x

				target_neighbors = torch.argsort(context_similarity, descending=True)[:k]
				context = torch.mean(context_x[target_neighbors], dim=0) if len(target_neighbors)>0 else 0.
				output_x[i] -= context

		else:
			# In test and validation phase, if no cheating,
			# we cannot access real targets so we just compute the mean on the movie
			# However I keep the selective aspect

			normalized_x = F.normalize(x, dim=-1, p=2)  # [N, D]
			similarity = normalized_x @ normalized_x.T
			for i in range(len(x)):
				id = ids[i]
				# First we select the x rows of the appropriate id and label
				context_booleans = ids == id
				context_x = x[context_booleans]
				# Then we consider similarity on those elements
				context_similarity = similarity[i, context_booleans]  # as many elements as context_x

				target_neighbors = torch.argsort(context_similarity, descending=True)[:k]
				context = torch.mean(context_x[target_neighbors], dim=0) if len(target_neighbors) > 0 else 0.
				output_x[i] -= context

		if self.detach_representation:
			repr = output_x.detach().clone()
		else:
			repr = output_x.clone()

		output_x = self.to_latent(output_x)

		if self.return_representation:
			return repr, self.mlp_head(output_x)
		else:
			return self.mlp_head(output_x)


class DoubleDifferentialMLP(MLP):
	def __init__(self, feature_dim, dim, num_classes, representation_architecture, dropout=0., return_representation=False,
			detach_representation=True):
		super().__init__(feature_dim=feature_dim, dim=dim, num_classes=num_classes, dropout=dropout,
						 return_representation=return_representation, representation_architecture=representation_architecture,
						 detach_representation=detach_representation)

	def forward(self, x, ids, binary_targets):

		x = self.representation_mlp(x)
		output_x = x.clone()  # we will modify the rows from this tensor

		if binary_targets is not None:
			for id in torch.unique(ids):
				# We compute the representation of the contexts
				negative_context_booleans = torch.bitwise_and(ids == id, binary_targets == 0)
				id_negative_context = torch.mean(x[negative_context_booleans], dim=0) if torch.any(
					negative_context_booleans) else 0.
				positive_context_booleans = torch.bitwise_and(ids == id, binary_targets == 1)
				id_positive_context = torch.mean(x[positive_context_booleans], dim=0) if torch.any(
					positive_context_booleans) else 0.
				# We compute x representation relatively to those contexts
				output_x[torch.bitwise_and(ids == id, binary_targets == 1)] -= id_negative_context
				output_x[torch.bitwise_and(ids == id, binary_targets == 0)] -= id_positive_context

		else:
			print(f"Using neutral context (because no pseudo-labels yet)...")
			# In test and validation phase, if no cheating,
			# we cannot access real targets so we just compute the mean on the movie
			for id in torch.unique(ids):
				id_context = torch.mean(x[ids == id], dim=0)
				output_x[ids == id] -= id_context

		if self.detach_representation:
			repr = output_x.detach().clone()
		else:
			repr = output_x.clone()

		output_x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(output_x)
		else:
			return self.mlp_head(output_x)


class ScheduledDoubleDifferentialMLP(MLP):
	def __init__(self, feature_dim, dim, num_classes, representation_architecture, dropout=0., return_representation=False,
			detach_representation=True):
		super().__init__(feature_dim=feature_dim, dim=dim, num_classes=num_classes, dropout=dropout,
						 return_representation=return_representation, representation_architecture=representation_architecture,
						 detach_representation=detach_representation)

	def forward(self, x, ids, binary_targets, sampling_rate=0.):

		x = self.representation_mlp(x)
		output_x = x.clone()  # we will modify the rows from this tensor

		# Here we do the differential operation
		device = x.get_device()
		sampled_indices = (torch.rand(len(x)) < sampling_rate).to(device) # a boolean array indicating, for each obs, whether it is "sampled", ie. if we do apply double differential
		for id in torch.unique(ids):
			# We compute the representation of the contexts
			negative_context_booleans = torch.bitwise_and(ids == id, binary_targets == 0)
			id_negative_context = torch.mean(x[negative_context_booleans], dim=0) if torch.any(negative_context_booleans) else 0.
			positive_context_booleans = torch.bitwise_and(ids == id, binary_targets == 1)
			id_positive_context = torch.mean(x[positive_context_booleans], dim=0) if torch.any(positive_context_booleans) else 0.
			id_global_context = torch.mean(x[ids == id], dim=0)

			# For sampling_rate of examples, we apply double differential ; for the others we apply standard differential
			output_x[torch.bitwise_and(sampled_indices, torch.bitwise_and(ids == id, binary_targets == 1))] -= id_negative_context
			output_x[torch.bitwise_and(sampled_indices, torch.bitwise_and(ids == id, binary_targets == 0))] -= id_positive_context
			output_x[torch.bitwise_and(torch.bitwise_not(sampled_indices), ids == id)] -= id_global_context

		if self.detach_representation:
			repr = output_x.detach().clone()
		else:
			repr = output_x.clone()

		output_x = self.to_latent(output_x)

		if self.return_representation:
			return repr, self.mlp_head(output_x)
		else:
			return self.mlp_head(output_x)


class FrozenContextDifferentialMLP(MLP):
	def __init__(self, feature_dim, dim, num_classes, representation_architecture, dropout=0., return_representation=False,
			detach_representation=True, backbone=None):
		super().__init__(feature_dim=feature_dim, dim=dim, num_classes=num_classes, dropout=dropout,
						 return_representation=return_representation, representation_architecture=representation_architecture,
						 detach_representation=detach_representation)

		self.backbone = backbone # unlike aposteriori_differential, will only be used to compute the context
		# We freeze the weights of the backbone
		for param in self.backbone.parameters():
			param.requires_grad = False

	def forward(self, x, ids):

		context_x, context_logits = self.backbone(x)  # we must ensure that return_representation=True for the backbone

		x = self.representation_mlp(x)
		output_x = x.clone()  # we will modify the rows from this tensor

		# Here we do the differential operation
		for id in torch.unique(ids):
			# We compute the representation of the context
			id_context = torch.mean(context_x[ids == id], dim=0)
			# We compute x representation relatively to this context
			output_x[ids == id] -= id_context

		if self.detach_representation:
			repr = output_x.detach().clone()
		else:
			repr = output_x.clone()

		output_x = self.to_latent(output_x)

		if self.return_representation:
			return repr, self.mlp_head(output_x)
		else:
			return self.mlp_head(output_x)
