import torch

from models.cem_models.cem import ConceptEmbeddingModel
from models.vit_models import *


def latent_code_generator_model(output_dim):
    if output_dim is None:
        output_dim = 128
    return torch.nn.Sequential(*[
        # 4 because Dot has inputs with 4 features in them
        torch.nn.Linear(4, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, output_dim),
    ])


# We simply import our CEM class (the same can be done with CBMs to easily train
# any of their variants)

feature_dim=512
max_seq_length=480  # max seq len represents the maximum numero of 16 frames features
dim=128
depth=2  # transformer depth
heads=8
mlp_dim=1024
pool='cls-pooling'
dim_head=64
dropout=0.
emb_dropout=0.

representation_vit = RepresentationViT(
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
h_dim=128
# And generate the actual model
cem_model = ConceptEmbeddingModel(
  n_concepts=2, # Number of training-time concepts. Dot has 2
  n_tasks=1, # Number of output labels. Dot is binary so it has 1.
  emb_size=128,  # We will use an embedding size of 128
  concept_loss_weight=1,  # The weight assigned to the concept prediction loss relative to the task predictive loss.
  learning_rate=1e-3,  # The learning rate to use during training.
  optimizer="adam",  # The optimizer to use during training.
  training_intervention_prob=0.25, # RandInt probability. We recommend setting this to 0.25.
  c_extractor_module=representation_vit, # Here we provide our generating function for the latent code generator model.
	h_dim=h_dim,
  c2y_model=None,  # We will let the API simply add a linear layer from the concept bottleneck to the downstream task labels
)
print(cem_model)
