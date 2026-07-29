from torch.utils.data import Sampler
import numpy as np
from random import shuffle
from math import ceil


class DifferentialSampler(Sampler):
	def __init__(self, train_dataset, nb_clips_per_movie):
		super().__init__()
		self.ids = train_dataset.gt_df["movie_id"].unique().tolist()
		#print(f"self.ids:\n{self.ids}")
		self.indices_by_ids = {k: np.where(train_dataset.gt_df["movie_id"] == k)[0].tolist() for k in self.ids}
		dict_to_print = {k: len(v) for k, v in self.indices_by_ids.items()}
		#print(f"self.indices_by_ids : \n{dict_to_print}")
		#print(f"total number of clips : {np.sum([v for v in dict_to_print.values()])}")
		# N.B. : the indices the sampler will return are classical indices 0, 1 etc., they do not correspond
		# to pandas index (because the get_item method of Dataset uses iloc) ; hence the argsort
		self.nb_total_data = len(train_dataset)
		self.nb_clips_per_movie = nb_clips_per_movie
		# N.B. : in the following, we make the arbitrary choice to return only "full batches per movie",
		# so at each epoch we will ignore part of the clips of a given movie, if its number of clips is not a multiple
		# of nb_clips_per_movie
		nb_returned = 0
		nb_batch_returned = 0
		for list_indices in self.indices_by_ids.values():
			nb_returned += (len(list_indices) // self.nb_clips_per_movie) * self.nb_clips_per_movie
			nb_batch_returned += (len(list_indices) // self.nb_clips_per_movie)
		self.nb_returned = nb_returned
		#print(f"self.nb_returned : {self.nb_returned}")
		#print(f"nb_batch_returned : {nb_batch_returned}")

	def __len__(self):
		# According to documentation, "returns the length of the returned iterators"
		return self.nb_returned

	def __iter__(self):
		# First, we create all of our batches
		batches = []
		for id, list_indices in self.indices_by_ids.items():
			# a) we shuffle the ids of a given movie in order to introduce some randomness
			shuffle(list_indices) # note that it does modify the actual value of the dict
			# b) we create the batches associated to this movie
			for i_batch in range((len(list_indices) // self.nb_clips_per_movie)):
				batch = list_indices[i_batch*self.nb_clips_per_movie:(i_batch+1)*self.nb_clips_per_movie]
				# c) we append them to the list of movie-batches
				batches.append(batch)
		# d) We shuffle the list of batches, so that the order of movie-batches is not always the same
		shuffle(batches)
		# e) we flatten it
		returned_indices = [i for batch in batches for i in batch]
		#print(f"len(returned_indices) : {len(returned_indices)}")
		yield from returned_indices


class FullMovieDifferentialSampler(Sampler):
	def __init__(self, train_dataset, nb_movies_per_batch=1):
		super().__init__()
		self.ids = train_dataset.gt_df["movie_id"].unique().tolist()
		self.indices_by_ids = {k: np.where(train_dataset.gt_df["movie_id"] == k)[0].tolist() for k in self.ids}
		self.nb_total_data = len(train_dataset)
		self.nb_movies_per_batch = nb_movies_per_batch

	def __len__(self):
		# According to documentation, "returns the length of the returned iterators"
		return self.nb_total_data

	def __iter__(self):
		# a) we shuffle the movies, so that they are not always in the same order
		shuffle(self.ids)
		# b) we return the batches, batch of nb_movies by batch of nb_movies
		for i_batch in range(ceil(len(self.ids) / self.nb_movies_per_batch)):
			batch_ids = self.ids[i_batch*self.nb_movies_per_batch:(i_batch+1)*self.nb_movies_per_batch]
			batch = [i for batch_id in batch_ids for i in self.indices_by_ids[batch_id]]
			yield batch


class TemporalDifferentialSampler(Sampler):
	# Like a DifferentialSampler, except that in the movie clips from a batch you want them to be consecutive
	def __init__(self, train_dataset, nb_clips_per_movie):
		super().__init__()
		self.ids = train_dataset.gt_df["movie_id"].unique().tolist()
		#print(f"self.ids:\n{self.ids}")
		self.indices_by_ids = {k: np.where(train_dataset.gt_df["movie_id"] == k)[0].tolist() for k in self.ids}
		# Note that, because the datasets are ordered, the lists in the dict should be ordered as well
		dict_to_print = {k: len(v) for k, v in self.indices_by_ids.items()}
		#print(f"self.indices_by_ids : \n{dict_to_print}")
		#print(f"total number of clips : {np.sum([v for v in dict_to_print.values()])}")
		# N.B. : the indices the sampler will return are classical indices 0, 1 etc., they do not correspond
		# to pandas index (because the get_item method of Dataset uses iloc) ; hence the argsort
		self.nb_total_data = len(train_dataset)
		self.nb_clips_per_movie = nb_clips_per_movie
		# N.B. : in the following, we make the arbitrary choice to return only "full batches per movie",
		# so at each epoch we will ignore part of the clips of a given movie, if its number of clips is not a multiple
		# of nb_clips_per_movie
		nb_returned = 0
		nb_batch_returned = 0
		for list_indices in self.indices_by_ids.values():
			nb_returned += (len(list_indices) // self.nb_clips_per_movie) * self.nb_clips_per_movie
			nb_batch_returned += (len(list_indices) // self.nb_clips_per_movie)
		self.nb_returned = nb_returned
		#print(f"self.nb_returned : {self.nb_returned}")
		#print(f"nb_batch_returned : {nb_batch_returned}")

	def __len__(self):
		# According to documentation, "returns the length of the returned iterators"
		return self.nb_returned

	def __iter__(self):
		# First, we create all of our batches
		batches = []
		for id, l in self.indices_by_ids.items():
			# a) we do not shuffle the ids of a given movie in order to preserve temporal order :
			# this is the only difference with DifferentialSampler
			# N.B. : because of only taking full batches, the end of the movies could systematically be ignored
			# So we introduce a random translation
			translation_range = len(l) % self.nb_clips_per_movie
			translation = np.random.randint(low=0, high=translation_range+1)
			list_indices = l[translation:]
			# b) we create the batches associated to this movie
			for i_batch in range((len(list_indices) // self.nb_clips_per_movie)):
				batch = list_indices[i_batch*self.nb_clips_per_movie:(i_batch+1)*self.nb_clips_per_movie]
				# c) we append them to the list of movie-batches
				batches.append(batch)
		# d) We shuffle the list of batches, so that the order of movie-batches is not always the same
		shuffle(batches)
		# e) we flatten it
		returned_indices = [i for batch in batches for i in batch]
		#print(f"len(returned_indices) : {len(returned_indices)}")
		yield from returned_indices
