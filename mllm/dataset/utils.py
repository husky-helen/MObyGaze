import torch
import numpy as np
import cv2

from constants import MODALITY_TO_CONCEPTS


def compress_frame(frame,
                   max_pixels,
                   factor,
				   strict_compress=False):
	if strict_compress:
		# we want to enforce the new frame definition
		# (useful when dealing with different sources to uniformize their shape)
		new_h = factor
		new_w = max_pixels // factor
		return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
	else:
		# We want to preserve the original scale
		h, w = frame.shape[:2]

		if (h * w <= max_pixels) and (h % factor == 0) and (w % factor == 0):
			return frame
		scale = (max_pixels / (h * w)) ** 0.5
		new_h = max(int((h * scale) // factor) * factor, factor)
		new_w = max(int((w * scale) // factor) * factor, factor)
		return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sample_video(path: str, start_frame, end_frame, num_frames,
				 max_pixels, factor, strict_compress=False):
	cap = cv2.VideoCapture(path)
	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# total is the detected number of frames, based on metadata :
	# in practice, for our data, it matches the actual number of frames of the video file
	# So, there is no need to fear ret=False in the range indicated by our annotation files
	if total == 0:
		raise RuntimeError(f"Cannot read video: {path}")

	idxs = np.linspace(start_frame, end_frame, num_frames, endpoint=False, dtype=int)
	frames = []

	for frame_id in idxs:
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
		ret, frame = cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = compress_frame(frame, max_pixels=max_pixels, factor=factor, strict_compress=strict_compress)
		frames.append(frame)
		# To plot the frame :
		#plt.imshow(frame)
		#images_dir = f"{PROJECT_DIR}/images"
		#plt.savefig(f"{images_dir}/frame_id_{frame_id}.png")
	cap.release()
	if len(frames) != num_frames:
		print(f"Warning: obtained {len(frames)} frames VS {num_frames} specified num_frames")
	return np.stack(frames)


def get_frames_of_video(path: str, start_frame, end_frame, num_frames, max_pixels=None, factor=None):
	cap = cv2.VideoCapture(path)
	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# total is the detected number of frames, based on metadata :
	# in practice, for our data, it matches the actual number of frames of the video file
	# So, there is no need to fear ret=False in the range indicated by our annotation files
	if total == 0:
		raise RuntimeError(f"Cannot read video: {path}")

	idxs = np.linspace(start_frame, end_frame, num_frames, endpoint=False, dtype=int)
	frames = []

	for frame_id in idxs:
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
		ret, frame = cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		if max_pixels is not None and factor is not None:
			frame = compress_frame(frame, max_pixels=max_pixels, factor=factor)

		frames.append(frame)
		# To plot the frame :
		#plt.imshow(frame)
		#images_dir = f"{PROJECT_DIR}/images"
		#plt.savefig(f"{images_dir}/frame_id_{frame_id}.png")
	cap.release()
	if len(frames) != num_frames:
		print(f"Warning: obtained {len(frames)} frames VS {num_frames} specified num_frames")
	return frames


def listconcepts2onehot(list_concepts, modality="visual"):
	target_concepts = MODALITY_TO_CONCEPTS[modality]
	res = torch.zeros(len(target_concepts))
	for i, word in enumerate(target_concepts):
		if word in list_concepts:
			res[i] = 1

	return res
