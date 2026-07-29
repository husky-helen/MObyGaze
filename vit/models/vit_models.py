import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from copy import deepcopy


# ViT code is inspired from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vivit.py
def exists(val):
	return val is not None


def pair(t):
	return t if isinstance(t, tuple) else (t, t)


def weight_init_final(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1 or classname.find('Linear') != -1:
		torch_init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0)


# Global Attention: The self-attention mechanism allows each token to attend to all other tokens,
# regardless of their position in time. This enables the model to capture long-range temporal dependencies.
# No Explicit Temporal Modeling: Unlike some architectures that use 3D convolutions or recurrent layers,
# this model relies entirely on the self-attention mechanism to capture temporal relationships.

class FeedForward(
	nn.Module):  # Transformer Encoder MLP layer - two layer with GELU - https://viso.ai/deep-learning/vision-transformer-vit/
	def __init__(self, dim, hidden_dim, dropout=0.):
		super().__init__()
		self.net = nn.Sequential(
			nn.LayerNorm(dim),
			nn.Linear(dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, dim),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		return self.net(x)


class Attention(nn.Module):  # Transfomer Encoder Multi Head Self Attention (MSP)
	def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
		super().__init__()
		inner_dim = dim_head * heads
		project_out = not (heads == 1 and dim_head == dim)

		self.heads = heads
		self.scale = dim_head ** -0.5

		self.norm = nn.LayerNorm(dim)
		self.attend = nn.Softmax(dim=-1)
		self.dropout = nn.Dropout(dropout)

		self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

		self.to_out = nn.Sequential(
			nn.Linear(inner_dim, dim),
			nn.Dropout(dropout)
		) if project_out else nn.Identity()

	def forward(self, x, mask=None):

		# print(f"Attention input shape: {x.shape}")
		# if mask is not None:
		#    print(f"Attention mask shape: {mask.shape}")

		x = self.norm(x)
		qkv = self.to_qkv(x).chunk(3, dim=-1)
		q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

		dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

		##IT WAS HERE HERE!!! --> wrong order it needs to be applied after the mask because otherwise calculates the attention probabilities without considering the mask!
		# attn = self.attend(dots)
		# attn = self.dropout(attn)

		# Before applying softmax, we use the mask to set attention scores for padded positions to negative infinity - This ensures that padded positions will have zero attention weight after softmax.
		if mask is not None:
			if x.size(1) > mask.size(1):  # If `x` includes CLS token
				# expand mask for all heads and adjust for CLS token
				mask = mask[:, None, None, :]
				mask = F.pad(mask, (1, 0), value=1)  # add 1 column at the start for CLS token

			else:
				mask = mask[:, None, None, :]

			mask = mask.expand(-1, self.heads, dots.size(2), -1)

			# print(f"Expanded mask shape: {mask.shape}")
			# print(f"Dots shape: {dots.shape}")

			assert mask.shape == dots.shape, f"Mask shape {mask.shape} doesn't match dots shape {dots.shape}"
			dots = dots.masked_fill(mask == 0, float('-inf'))

		attn = self.attend(dots)
		attn = self.dropout(attn)

		out = torch.matmul(attn, v)
		out = rearrange(out, 'b h n d -> b n (h d)')
		return self.to_out(out)


# Image transfomer basic https://viso.ai/deep-learning/vision-transformer-vit/
class Transformer(
	nn.Module):  # Image of transfomer encoder all transfomer encoder blocks - Norm (inside attention) + Multi Head Attention + Res Connect + Norm (inside forward) + MLP
	def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
		super().__init__()
		self.norm = nn.LayerNorm(dim)
		self.layers = nn.ModuleList([])
		for _ in range(depth):  # depth is the number of blocks
			self.layers.append(nn.ModuleList([
				Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
				FeedForward(dim, mlp_dim, dropout=dropout)
			]))

	def forward(self, x, mask=None):
		for attn, ff in self.layers:
			x = attn(x, mask) + x  # attention wtih mask + residual connection
			x = ff(x) + x
		return self.norm(x)


class RepresentationViT(nn.Module):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.
	):
		super().__init__()

		assert pool in {'cls-pooling', 'avg-pooling'}, 'pool type must be either cls-pooling (cls token) or avg-pooling (mean pooling)'

		self.feature_dim = feature_dim
		self.max_seq_length = max_seq_length

		self.to_patch_embedding = nn.Sequential(
			nn.LayerNorm(feature_dim),  # normalisation before patch embeddings?
			nn.Linear(feature_dim, dim),
			nn.LayerNorm(dim)
		)

		self.pos_embedding = nn.Parameter(
			torch.randn(1, max_seq_length, dim))  # num frames only used for pos embeddibng? and the attntion mask?
		self.dropout = nn.Dropout(emb_dropout)

		self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) if pool == 'cls-pooling' else None

		self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

		self.pool = pool

	def forward(self, x, mask):

		b, t, f = x.shape  # batch, time, feature

		assert t == self.max_seq_length, f'Input frames {t} doesn\'t match model frames {self.max_seq_length}'
		assert f == self.feature_dim, f'Input feature dim {f} doesn\'t match model feature dim {self.feature_dim}'

		x = self.to_patch_embedding(x)

		if self.pool == 'cls-pooling':
			x += self.pos_embedding[:, :(
					t + 1)]  # slices the positional embedding to match the length of the input sequence plus one (or the CLS token)
		else:
			x += self.pos_embedding[:, :t]  # for average pooling

		x = self.dropout(x)

		if self.cls_token is not None:
			cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
			x = torch.cat((cls_tokens, x), dim=1)
		# print(f"Shape after adding CLS token: {x.shape}")

		x = self.transformer(x, mask)

		if self.pool == 'avg-pooling':
			x = x.mean(dim=1)
		else:
			x = x[:, 0]

		return x


class ViT(nn.Module):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True
			# usually, as I only return the representation as a debugging/analysis feature, I can detach it
			# In other cases however (only CEM, for now), this representation is used in the computation,
			# thus gradient descent, and should not be detached
	):
		super().__init__()
		self.representation_vit = RepresentationViT(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout)

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


	def forward(self, x, mask):

		x = self.representation_vit(x, mask)
		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()
		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class MultiModalViT(nn.Module):
	def __init__(
			self,
			*,
			feature_dims,
			max_seq_lengths,
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True
	):
		super().__init__()

		self.nb_modalities = len(feature_dims)
		self.common_feature_dim = min(feature_dims)
		self.max_seq_length = sum(max_seq_lengths)

		projection_layers = dict()
		for id_modality in range(self.nb_modalities):
			projection_layers[f"modality_{id_modality}"] = torch.nn.Linear(feature_dims[id_modality], self.common_feature_dim)
		self.projection_layers = nn.ModuleDict(projection_layers)

		self.representation_vit = RepresentationViT(
			feature_dim=self.common_feature_dim,
			max_seq_length=self.max_seq_length,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout)

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


	def forward(self, xs, masks):
		projected_xs = [F.relu(self.projection_layers[f"modality_{id_modality}"](x_mod)) for id_modality, x_mod in enumerate(xs)]
		x = torch.cat(projected_xs, dim=1) # we want to concatenate the tensors along the temporal = token dimension
		mask = torch.cat(masks, dim=-1)

		x = self.representation_vit(x, mask)
		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()
		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class DifferentialViT(ViT):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True
	):
		super().__init__(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			num_classes=num_classes,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout,
			return_representation=return_representation,
			detach_representation=detach_representation)

	def forward(self, x, mask, ids):

		x = self.representation_vit(x, mask)

		# Here we do the differential operation
		for id in torch.unique(ids):
			# We compute the representation of the context
			id_context = torch.mean(x[ids == id], dim=0)
			# We compute x representation relatively to this context
			x[ids == id] -= id_context # see ComplexDifferentialViT to combine x and id_context in a more elaborate way

		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()

		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class GatedDifferentialViT(ViT):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True
	):
		super().__init__(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			num_classes=num_classes,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout,
			return_representation=return_representation,
			detach_representation=detach_representation)

		self.gating_vit = torch.nn.Linear(dim, 1)

	def forward(self, x, mask, ids):

		x = self.representation_vit(x, mask)
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


class ComplexGatedDifferentialViT(ViT):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True
	):
		super().__init__(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			num_classes=num_classes,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout,
			return_representation=return_representation,
		detach_representation=detach_representation)

		self.gating_vit = torch.nn.Linear(2*dim, 1)

	def forward(self, x, mask, ids):

		x = self.representation_vit(x, mask)

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


class AttentionLikeGatedDifferentialViT(ViT):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True
	):
		super().__init__(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			num_classes=num_classes,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout,
			return_representation=return_representation,
		detach_representation=detach_representation)

		self.mlp_1 = torch.nn.Linear(dim, dim // 2)
		self.mlp_2 = torch.nn.Linear(dim, dim // 2)

	def forward(self, x, mask, ids):

		x = self.representation_vit(x, mask)

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


class APosterioriDifferentialViT(nn.Module):
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

	def forward(self, x, mask, ids):

		x, logits = self.backbone(x, mask) # we must ensure that return_representation=True for the backbone

		# Here we do the differential operation
		for id in torch.unique(ids):
			# We compute the representation of the context
			id_context = torch.mean(x[ids == id], dim=0)
			# We compute x representation relatively to this context
			x[ids == id] -= id_context

		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()

		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class ComplexDifferentialViT(ViT):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True
	):
		super().__init__(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			num_classes=num_classes,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout,
			return_representation=return_representation,
			detach_representation=detach_representation)

		self.complex_mlp = nn.Sequential(
			nn.Linear(2*dim, dim),
			nn.BatchNorm1d(dim),
			nn.ReLU(),
			nn.Dropout(dropout)
		)

	def forward(self, x, mask, ids):

		x = self.representation_vit(x, mask)

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
class DifferentialViTNegative(ViT):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True
	):
		super().__init__(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			num_classes=num_classes,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout,
			return_representation=return_representation,
			detach_representation=detach_representation)

	def forward(self, x, mask, ids, binary_targets):

		x = self.representation_vit(x, mask)

		# Here we do the differential operation
		if binary_targets is not None:
			for id in torch.unique(ids):
				# We compute the representation of the context
				context_booleans = torch.bitwise_and(ids == id, binary_targets == 0)
				id_context = torch.mean(x[context_booleans], dim=0) if torch.any(context_booleans) else 0.
				# We compute x representation relatively to this context
				x[ids == id] -= id_context # in the future, we ay want to combine x and id_context in a more elaborate way
		else:
			# In test and validation phase, if no cheating,
			# we cannot access real targets so we just compute the mean on the movie
			for id in torch.unique(ids):
				id_context = torch.mean(x[ids == id], dim=0)
				x[ids == id] -= id_context

		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()

		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class DifferentialViTSelective(ViT):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True
	):
		super().__init__(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			num_classes=num_classes,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout,
			return_representation=return_representation,
			detach_representation=detach_representation)

	def forward(self, x, mask, ids):

		k = 8 # number of closest neighbors we consider

		x = self.representation_vit(x, mask)
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


class DifferentialViTSelectiveNegative(ViT):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True
	):
		super().__init__(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			num_classes=num_classes,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout,
			return_representation=return_representation,
			detach_representation=detach_representation)

	def forward(self, x, mask, ids, binary_targets):

		k = 8 # number of closest neighbors we consider

		x = self.representation_vit(x, mask)
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


class DoubleDifferentialViT(ViT):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True
	):
		super().__init__(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			num_classes=num_classes,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout,
			return_representation=return_representation,
			detach_representation=detach_representation)

	def forward(self, x, mask, ids, binary_targets):

		x = self.representation_vit(x, mask)

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
				x[torch.bitwise_and(ids == id, binary_targets == 1)] -= id_negative_context
				x[torch.bitwise_and(ids == id, binary_targets == 0)] -= id_positive_context

		else:
			print(f"Using neutral context (because no pseudo-labels yet)...")
			# In test and validation phase, if no cheating,
			# we cannot access real targets so we just compute the mean on the movie
			for id in torch.unique(ids):
				id_context = torch.mean(x[ids == id], dim=0)
				x[ids == id] -= id_context

		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()

		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class ScheduledDoubleDifferentialViT(ViT):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True
	):
		super().__init__(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			num_classes=num_classes,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout,
			return_representation=return_representation,
			detach_representation=detach_representation)

	def forward(self, x, mask, ids, binary_targets, sampling_rate=0.):

		x = self.representation_vit(x, mask)

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
			x[torch.bitwise_and(sampled_indices, torch.bitwise_and(ids == id, binary_targets == 1))] -= id_negative_context
			x[torch.bitwise_and(sampled_indices, torch.bitwise_and(ids == id, binary_targets == 0))] -= id_positive_context
			x[torch.bitwise_and(torch.bitwise_not(sampled_indices), ids == id)] -= id_global_context

		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()

		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class FrozenContextDifferentialViT(ViT):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True,
			backbone=None
	):
		super().__init__(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			num_classes=num_classes,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout,
			return_representation=return_representation,
			detach_representation=detach_representation)

		self.backbone = backbone # unlike aposteriori_differential, will only be used to compute the context
		# We freeze the weights of the backbone
		for param in self.backbone.parameters():
			param.requires_grad = False

	def forward(self, x, mask, ids):

		context_x, context_logits = self.backbone(x, mask)  # we must ensure that return_representation=True for the backbone

		x = self.representation_vit(x, mask)

		# Here we do the differential operation
		for id in torch.unique(ids):
			# We compute the representation of the context
			id_context = torch.mean(context_x[ids == id], dim=0)
			# We compute x representation relatively to this context
			x[ids == id] -= id_context

		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()

		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class DifferentialViTIterative(ViT):
	def __init__(
			self,
			*,
			feature_dim=512,
			max_seq_length=480,  # max seq len represents the maximum numero of 16 frames features
			num_classes,
			dim,
			depth,  # transformer depth
			heads,
			mlp_dim,
			pool='cls-pooling',
			dim_head=64,
			dropout=0.,
			emb_dropout=0.,
			return_representation=False,
			detach_representation=True,
			num_iterations = 1 # number of iterations before returning the actual prediction
	):
		super().__init__(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			num_classes=num_classes,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout,
			return_representation=return_representation,
			detach_representation=detach_representation)
		self.num_iterations = num_iterations

	def forward(self, x, mask, ids):

		x = self.representation_vit(x, mask)

		# Here we do the differential operation
		for i in range(self.num_iterations):
			binary_targets = (torch.sigmoid(self.mlp_head(x)) >= 0.5).int().squeeze()
			for id in torch.unique(ids):
				# We compute the representation of the context
				context_booleans = torch.bitwise_and(ids == id, binary_targets == 0)
				id_context = torch.mean(x[context_booleans], dim=0) if torch.any(context_booleans) else 0.
				# We compute x representation relatively to this context
				x[ids == id] -= id_context

		if self.detach_representation:
			repr = x.detach().clone()
		else:
			repr = x.clone()

		x = self.to_latent(x)

		if self.return_representation:
			return repr, self.mlp_head(x)
		else:
			return self.mlp_head(x)


class RepresentationDifferentialViT(nn.Module):
	def __init__(
			self,
			*,
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
	):
		super().__init__()

		self.representation_vit = RepresentationViT(
			feature_dim=feature_dim,
			max_seq_length=max_seq_length,
			dim=dim,
			depth=depth,
			heads=heads,
			mlp_dim=mlp_dim,
			pool=pool,
			dim_head=dim_head,
			dropout=dropout,
			emb_dropout=emb_dropout)

	def forward(self, x, mask, ids):
		x = self.representation_vit(x, mask)

		# Here we do the differential operation
		for id in torch.unique(ids):
			# We compute the representation of the context
			id_context = torch.mean(x[ids == id], dim=0)
			# We compute x representation relatively to this context
			x[ids == id] -= id_context # see ComplexDifferentialViT to combine x and id_context in a more elaborate way

		return x


# In the two-stage approach, we define the classification head separately from the representation head,
# as they will be trained separately
# N.B. : Even if they are trained separately (in terms of back-propagation),
# we can train them either successively (2-stage)
# or simultaneously (the successive approach should be slightly better, but slower)
# We define it in a similar fashion as the ViT classification head
class ClassificationHead(nn.Module):
	def __init__(
			self,
			*,
			dim,
			num_classes,
			dropout=0.,
	):
		super().__init__()

		self.mlp_head = nn.Sequential(  # FSL neurips experiment layer
			nn.Linear(dim, dim // 2),
			nn.BatchNorm1d(dim // 2),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(dim // 2, num_classes)
		)

	def forward(self, x):
		return self.mlp_head(x)


class SimilarityBasedClassifier(nn.Module):
	def __init__(
			self,
			train_embeddings,
			train_labels,
			train_ids,
			beta,
			device,
			similarity_type,
			clustering_k

	):
		super().__init__()

		self.classes = torch.unique(train_labels, sorted=True)
		self.num_classes = len(self.classes)
		# where N is the number of train / "few-shot" examples characterizing this non-parametric model,
		# D is the mebedding dimension and C the number of classes

		self.cache_keys = F.normalize(train_embeddings, dim=-1, p=2).to(device)  # [N, D]
		self.cache_values = (F.one_hot(torch.Tensor(train_labels), num_classes=self.num_classes)).to(torch.float).to(
			device)
		# [N, C] ; train_labels is expected to be of shape [N]

		self.train_ids = train_ids.to(
			device)  # [N] ; may be useful in some cases for variations of the standard similarity-based classifier
		self.beta = beta

		self.similarity_type = similarity_type
		self.clustering_k = clustering_k
		self.device = device

	def set_beta(self, beta):
		self.beta = beta

	def set_clustering_k(self, k):
		self.clustering_k = k

	def forward(self, x):
		# x has shape [batch_size, D]
		# inspired by compute_cache_logits in https://github.com/kdiAAA/TDA/blob/main/tda_runner.py (Efficient Test-Time Adaptation of Vision-Language Models)
		# N.B. : there is no need of alpha here, as there is no need to balance the logits between two components
		# N.B. : to be consistent with the inspiring articles, we make sure both the cache and the input features are normalized
		with torch.no_grad():
			x = F.normalize(x, dim=-1, p=2).to(self.device)
			similarity = x @ self.cache_keys.T
			print(f"x min, max elements :\n{torch.min(x)}, {torch.max(x)}")
			print(f"cache_keys min, max elements :\n{torch.min(self.cache_keys)}, {torch.max(self.cache_keys)}")

			print(f"similarity.shape :\n{similarity.shape}")
			print(
				f"similarity min, max -> avg elements :\n{torch.min(similarity)}, {torch.max(similarity)} -> {torch.mean(similarity)}")
			affinity = ((-1) * (self.beta - self.beta * similarity)).exp().to(self.device)  # [batch_size, N]
			print(f"affinity.shape :\n{affinity.shape}")
			print(
				f"affinity min, max -> avg elements :\n{torch.min(affinity)}, {torch.max(affinity)} -> {torch.mean(affinity)}")

		if self.similarity_type == "similarity_clustering":
			# We zero out some affinities such that the result is equivalent to embracing diversity
			# use self.clustering_k
			for i, aff in enumerate(affinity):
				# as it's a complex manipulation, the easiest way is to go through the lines
				# a line corresponds to one test input
				ind_to_zero_out = []  # the indices we will zero out on this line
				for id in torch.unique(self.train_ids):
					id_affinities = aff[
						self.train_ids == id]  # the affinities with the clip from this movie/id = restricted aff vector
					id_ind = torch.arange(0, len(self.train_ids)).to(self.device)[
						self.train_ids == id]  # the indices of those affinities in the full aff vector
					furthest_from_id = torch.argsort(id_affinities, descending=True)[
						self.clustering_k:]  # the indices (in the restricted aff vector) of the elements to zero out
					ind_furthest_from_id = id_ind[
						furthest_from_id]  # the indices (in the full aff vector) of the elements to zero out
					ind_to_zero_out.append(ind_furthest_from_id)
				ind_to_zero_out = torch.cat(ind_to_zero_out)
				affinity[i, ind_to_zero_out] = 0.

		cache_logits = affinity @ self.cache_values  # [batch_size, C]
		print(f"cache_logits[:5] :\n{cache_logits[:5]}")
		return cache_logits  # because our core function expects only positive probabilities
