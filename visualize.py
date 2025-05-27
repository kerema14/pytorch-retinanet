import numpy as np

import time

import time
import argparse
import random


import cv2

import torch
from torch.utils.data import  DataLoader
from torchvision import  transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler,  \
	UnNormalizer, Normalizer


assert torch.__version__.split('.')[0] == '2'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--model', help='Path to model (.pt) file.')

	parser = parser.parse_args(args)

	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Resizer()]))
	elif parser.dataset == 'csv':
		dataset_val = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	retinanet = torch.load(parser.model)

	use_gpu = True

	if use_gpu:
		if torch.cuda.is_available():
			retinanet = retinanet.cuda()

	if torch.cuda.is_available():
		retinanet = torch.nn.DataParallel(retinanet).cuda()
	else:
		retinanet = torch.nn.DataParallel(retinanet)

	retinanet.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):
		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	def resize_image_maintain_aspect(img, target_size):
		"""Resize image while maintaining aspect ratio and pad with black if needed"""
		h, w = img.shape[:2]
		
		# Calculate scaling factor
		scale = min(target_size / w, target_size / h)
		new_w = int(w * scale)
		new_h = int(h * scale)
		
		# Resize image
		resized = cv2.resize(img, (new_w, new_h))
		
		# Create black canvas and center the resized image
		canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
		start_x = (target_size - new_w) // 2
		start_y = (target_size - new_h) // 2
		canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
		
		return canvas

	# Convert dataloader to list and shuffle for random selection
	all_data = list(enumerate(dataloader_val))
	random.shuffle(all_data)
	
	# Select 36 random images
	selected_data = all_data[:36]
	
	# Process selected images
	processed_images = []
	
	print("Processing 36 random images...")
	for i, (original_idx, data) in enumerate(selected_data):
		print(f"Processing image {i+1}/36...")
		
		with torch.no_grad():
			st = time.time()
			if torch.cuda.is_available():
				scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
			else:
				scores, classification, transformed_anchors = retinanet(data['img'].float())
			
			idxs = np.where(scores.cpu() > 0.5)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img[img < 0] = 0
			img[img > 255] = 255

			img = np.transpose(img, (1, 2, 0))
			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			# Draw detections
			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)
				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

			# Resize image to fixed size for grid
			img_resized = resize_image_maintain_aspect(img, 200)  # Each cell will be 200x200
			processed_images.append(img_resized)

	# Create 6x6 grid
	grid_size = 6
	cell_size = 200
	grid_img = np.zeros((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8)

	for i in range(36):
		row = i // grid_size
		col = i % grid_size
		start_y = row * cell_size
		end_y = start_y + cell_size
		start_x = col * cell_size
		end_x = start_x + cell_size
		
		grid_img[start_y:end_y, start_x:end_x] = processed_images[i]

	# Display the grid
	print("Displaying grid of 36 random images...")
	cv2.imshow('36 Random Images with Detections', grid_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Optionally save the grid image
	cv2.imwrite('detection_grid_36_images.jpg', grid_img)
	print("Grid image saved as 'detection_grid_36_images.jpg'")


if __name__ == '__main__':
	main()