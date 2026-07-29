import torch
import numpy as np
import json
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score

from constants import *

with open(f'{PROJECT_DIR}/util_data/id_to_title.json', 'r') as file:
	id_to_title = json.load(file)


def get_linear_clf(type_prober):
	if type_prober == "svm":
		clf = LinearSVC(random_state=42)
		scoring_clf = False
	elif type_prober == "logistic":
		clf = LogisticRegression(random_state=42)
		scoring_clf = True
	elif type_prober == 'knn':
		clf = KNeighborsClassifier(metric='cosine', weights='uniform', n_neighbors=5)
		scoring_clf = True
	elif type_prober == 'mlp':
		clf = MLPClassifier(random_state=42, hidden_layer_sizes=(64,), activation='relu', solver='adam', alpha=0.0001)
		# by analogy with the standard MLP head, expected input has dimension 128
		scoring_clf = True
	elif type_prober in ['prior', 'uniform']:
		clf = DummyClassifier(strategy=type_prober, random_state=42)
		# prior returns the most frequent class in the train set, uniform is uniform
		scoring_clf = True
	else:
		raise Exception(f"Unknown type_prober '{type_prober}'")

	return clf, scoring_clf


def test_representations(outputs, type_prober):
	feats = outputs['feats'].detach().cpu().numpy()  # feats shape: [B, D]
	labels = outputs['labels'].detach().cpu().numpy()  # labels shape: [B]
	ids = outputs['ids'].detach().cpu().numpy()  # cluster ids shape: [B]
	if len(labels.shape) > 1:  # typically given labels may have shape [B, 1], which is useful in some contexts but not here
		labels = labels.reshape(-1)

	unique_ids, id_counts = np.unique(ids, return_counts=True)

	all_results = dict()

	for k, n_k in zip(unique_ids, id_counts):
		x = feats[ids == k]
		y_true = labels[ids == k]
		clf, scoring_clf = get_linear_clf(type_prober)
		clf.fit(x, y_true)
		y_pred = clf.predict(x)
		y_scores = clf.predict_proba(x)[:, 1] if scoring_clf else np.nan

		accuracy = accuracy_score(y_true, y_pred)
		f1 = f1_score(y_true, y_pred)
		f1_negative = f1_score(y_true, y_pred, pos_label=0)
		f1_macro = f1_score(y_true, y_pred, average='macro')
		f1_w = f1_score(y_true, y_pred, average='weighted')
		precision = precision_score(y_true, y_pred, zero_division=0.0)
		recall = recall_score(y_true, y_pred)
		pr_auc = average_precision_score(y_true, y_scores) if scoring_clf else np.nan

		id_dict = {'accuracy': accuracy, f'precision': precision, f'recall': recall, f'pr_auc': pr_auc}
		all_results[id_to_title[str(int(k))]] = id_dict

		considered_metrics = id_dict.keys()  # to construct the average results dict

	# We also compute the mean of those per-cluster results
	average_results = {k: float(np.nanmean([d[k] for d in all_results.values()])) for k in considered_metrics}
	all_results['average'] = average_results

	# We also compute the metrics of the global classification
	clf, scoring_clf = get_linear_clf(type_prober)
	clf.fit(feats, labels)
	y_pred = clf.predict(feats)
	y_scores = clf.predict_proba(feats)[:, 1] if scoring_clf else np.nan

	global_accuracy = accuracy_score(labels, y_pred)
	global_precision = precision_score(labels, y_pred, zero_division=0.0)
	global_recall = recall_score(labels, y_pred)
	global_pr_auc = average_precision_score(labels, y_scores) if scoring_clf else np.nan
	global_dict = {'accuracy': global_accuracy, f'precision': global_precision, f'recall': global_recall,
				   f'pr_auc': global_pr_auc}
	all_results['global'] = global_dict

	# We round the results for more readability
	for name_dict, metrics_dict in all_results.items():
		for k, v in metrics_dict.items():
			all_results[name_dict][k] = round(float(v), 3)

	all_results['prober'] = type_prober

	return all_results


def test_representations_various_probers(outputs, type_probers):
	return [test_representations(outputs, type_prober) for type_prober in type_probers]
