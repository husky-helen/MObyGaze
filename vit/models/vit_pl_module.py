import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, \
	classification_report
from sklearn.metrics import auc, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import io
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as torch_init
import numpy as np
import time
import neptune


# from neptune_tensorboard import enable_tensorboard_logging


# for the dataloader there's also track files and so on
class ViTLightningModule(pl.LightningModule):
	def __init__(self, model, learning_rate=1e-4, warmup_steps=None):  # original was 1e-5
		super().__init__()
		self.model = model
		self.warmup_steps = warmup_steps
		self.loss_ce = nn.BCEWithLogitsLoss()
		self.learning_rate = learning_rate

		self.best_epoch = None
		self.best_val_metrics = [(float('inf'), None)] * 3

		self.val_total_positive = 0
		self.val_total_negative = 0
		self.test_total_positive = 0
		self.test_total_negative = 0

	# enable_tensorboard_logging(self.run)

	@staticmethod
	def replace_nan_with_minus_one(arr):
		arr[np.isnan(arr)] = -1
		return arr

	def calculate_tb_metrics(self, loss=None, binary_targets_np=None, pred_label_np=None, probabilities_np=None,
							 mode=None, is_on_step=True, is_on_epoch=False):
		metrics = {}

		"calculate and log tensorboard metrics"
		accuracy = accuracy_score(binary_targets_np, pred_label_np)
		precision = precision_score(binary_targets_np, pred_label_np, zero_division=0)
		recall = recall_score(binary_targets_np, pred_label_np, zero_division=0)
		f1 = f1_score(binary_targets_np, pred_label_np, zero_division=0)
		f1_weighted = f1_score(binary_targets_np, pred_label_np, average='weighted', zero_division=0)

		metrics['loss'] = loss
		metrics['accuracy'] = accuracy
		metrics['precision'] = precision
		metrics['recall'] = recall
		metrics['f1'] = f1
		metrics['f1_weighted'] = f1_weighted

		self.log(f"{mode}_loss", loss, prog_bar=True)
		self.log(f"{mode}_accuracy", accuracy, on_step=is_on_step, on_epoch=is_on_epoch, prog_bar=True)
		self.log(f"{mode}_f1", f1, on_step=is_on_step, on_epoch=is_on_epoch, prog_bar=True)
		self.log(f"{mode}_f1_weighted", f1_weighted, on_step=is_on_step, on_epoch=is_on_epoch, prog_bar=True)
		self.log(f"{mode}_precision", precision, on_step=is_on_step, on_epoch=is_on_epoch, prog_bar=True)
		self.log(f"{mode}_recall", recall, on_step=is_on_step, on_epoch=is_on_epoch, prog_bar=True)

		if mode != 'train':
			fpr, tpr, threshold = roc_curve(binary_targets_np, probabilities_np)

			fpr = self.replace_nan_with_minus_one(fpr)
			tpr = self.replace_nan_with_minus_one(tpr)

			rec_auc = auc(fpr, tpr) if len(fpr) > 1 and len(
				tpr) > 1 else -1  # at least two distinct points to calculate AUC

			precision_curve, recall_curve, _ = precision_recall_curve(binary_targets_np, probabilities_np)
			pr_auc = auc(recall_curve, precision_curve) if len(precision_curve) > 1 and len(recall_curve) > 1 else -1

			self.log(f"{mode}_roc_auc", rec_auc, on_step=is_on_step, on_epoch=is_on_epoch, prog_bar=True)
			self.log(f"{mode}_pr_auc", pr_auc, on_step=is_on_step, on_epoch=is_on_epoch, prog_bar=True)

			metrics['pr_auc'] = pr_auc
			metrics['roc_auc'] = rec_auc

		return metrics

	def training_step(self, batch, batch_idx):
		visual_features, visual_mask, clip_label, one_hot_concepts = batch

		# print("VISUAL FEATURES")
		# print(visual_features)

		logits = self.model(visual_features, visual_mask)

		# print(logits)
		binary_targets = clip_label.float().unsqueeze(1)

		loss = self.loss_ce(logits, binary_targets)  # outputs logits

		probabilities = torch.sigmoid(logits)

		pred_label = (probabilities >= 0.5).int()

		binary_targets_np = binary_targets.detach().cpu().numpy()
		pred_label_np = pred_label.detach().cpu().numpy()

		metrics = self.calculate_tb_metrics(loss.item(), binary_targets_np, pred_label_np, mode='train',
											is_on_step=True, is_on_epoch=False)

		return {'loss': loss, 'metrics': metrics}

	def on_train_epoch_start(self):
		self.epoch_start_time = time.time()

	def on_train_epoch_end(self):
		epoch_duration = time.time() - self.epoch_start_time
		self.log('train_epoch_time', epoch_duration)
		print(f"Epoch {self.current_epoch + 1}, Duration: {epoch_duration:.2f} seconds")

		best_epochs = [f"Epoch {epoch} with val_loss: {loss:.4f}" for loss, epoch in self.best_val_metrics if
					   epoch is not None]
		print("Best models found at: ", ", ".join(best_epochs))
		for i, (loss, epoch) in enumerate(self.best_val_metrics):
			if epoch is not None:
				self.log(f'best_epoch_{i + 1}', epoch)
				self.log(f'best_val_loss_{i + 1}', loss)

	def validation_step(self, batch, batch_idx):
		visual_features, visual_mask, clip_label, one_hot_concepts = batch

		logits = self.model(visual_features, visual_mask)

		binary_targets = clip_label.float().unsqueeze(1)

		self.val_total_positive += torch.sum(binary_targets == 1).item()
		self.val_total_negative += torch.sum(binary_targets == 0).item()

		loss = self.loss_ce(logits, binary_targets)  # loss function expects logits then applies the inner sigmoid layer

		self.log(f'val_loss', loss, on_step=True, on_epoch=True)

		probabilities = torch.sigmoid(logits)

		pred_label = (probabilities >= 0.5).int()

		binary_targets_np = binary_targets.detach().cpu().numpy().astype(int)
		pred_label_np = pred_label.detach().cpu().numpy()
		probabilities_np = probabilities.detach().cpu().numpy()

		metrics = self.calculate_tb_metrics(loss.item(), binary_targets_np, pred_label_np, probabilities_np,
											mode='validation', is_on_step=True, is_on_epoch=False)

		if loss < max(self.best_val_metrics, key=lambda x: x[0])[0]:
			self.best_val_metrics.sort(key=lambda x: x[0], reverse=True)
			self.best_val_metrics[0] = (loss, self.current_epoch)
			self.best_val_metrics.sort(key=lambda x: x[0])
			print(f"New top model found at epoch {self.current_epoch} with val_loss: {loss:.4f}")

		return {'loss': loss, 'metrics': metrics}

	def on_validation_epoch_end(self):
		self.log("val_total_positive", self.val_total_positive, on_epoch=True, prog_bar=True)
		self.log("val_total_negative", self.val_total_negative, on_epoch=True, prog_bar=True)

		print(f"Validation - Total Positive: {self.val_total_positive}, Total Negative: {self.val_total_negative}")

		self.val_total_positive = 0
		self.val_total_negative = 0

	def test_step(self, batch, batch_idx):
		visual_features, visual_mask, clip_label, one_hot_concepts = batch

		logits = self.model(visual_features, visual_mask)

		binary_targets = clip_label.float().unsqueeze(1)

		self.test_total_positive += torch.sum(binary_targets == 1).item()
		self.test_total_negative += torch.sum(binary_targets == 0).item()

		loss = self.loss_ce(logits, binary_targets)

		probabilities = torch.sigmoid(logits)

		pred_label = (probabilities >= 0.5).int()

		binary_targets_np = binary_targets.detach().cpu().numpy().astype(int)
		pred_label_np = pred_label.detach().cpu().numpy()
		probabilities_np = probabilities.detach().cpu().numpy()

		metrics = self.calculate_tb_metrics(loss.item(), binary_targets_np, pred_label_np, probabilities_np,
											mode='test', is_on_step=False, is_on_epoch=True)

		# for metric_name, metric_value in metrics.items():
		#    if metric_name != 'loss':
		#        self.log(f'test_{metric_name}', metric_value, on_step=False, on_epoch=True)

		return {'loss': loss, 'metrics': metrics}

	def on_test_epoch_end(self):
		print(f"Test - Total Positive: {self.test_total_positive}, Total Negative: {self.test_total_negative}")
		self.log("test_total_positive", self.test_total_positive, on_epoch=True, prog_bar=True)
		self.log("test_total_negative", self.test_total_negative, on_epoch=True, prog_bar=True)

		self.test_total_positive = 0
		self.test_total_negative = 0

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.00005)
		scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)
		return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

	def optimizer_step(
			self, epoch, batch_idx, optimizer, optimizer_closure,
			on_tpu=False, using_native_amp=False, using_lbfgs=False):

		# print(f"global_step : {self.global_step} ; current lr : {optimizer.param_groups[0]['lr']}")

		optimizer.step(closure=optimizer_closure)

		if self.trainer.global_step < self.warmup_steps:
			# print("optimizer step called...reducing LR")
			lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup_steps)

			min_lr = 1e-7  # min lr threshold
			scaled_lr = max(min_lr, lr_scale * self.learning_rate)

			for pg in optimizer.param_groups:
				pg['lr'] = scaled_lr

		optimizer.zero_grad()

	def predict_step(self, batch, batch_idx):
		visual_features, visual_mask, clip_label, one_hot_concepts = batch

		logits = self.model(visual_features, visual_mask)

		probabilities = torch.sigmoid(logits)
		return probabilities
