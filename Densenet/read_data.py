# -*- coding: UTF-8 -*-

import tensorflow as tf 
import numpy as np
import os
import csv
import random
import json
import scipy.ndimage
import pickle

import augmentation

senior_path = '/DATA/data/yjgu/bladder/dwi_ax_preprocessed_2d_fixed_order'

def lists2slices(tuple_list):
	return [slice(*t) for t in tuple_list]

def read_bladder_bbox(json_file):
	with open(json_file, 'rb') as fd:
		bbox = lists2slices(json.load(fd))
	bbox = bbox[0:2]
	return bbox

def resize(image, new_shape):
	resize_factor = []
	for i, s in enumerate(new_shape):

		if s is None:
			resize_factor.append(1)
		else:
			resize_factor.append((s + 1e-3) / image.shape[i])
	# resize_factor = (np.round(new_shape).astype(np.float) + 1e-3) / image.shape
	# +1e-3 to suppress warning of scipy.ndimage.zoom
	new_image = scipy.ndimage.zoom(image, resize_factor, order=1)
	return new_image

def process_one_channel(image, bladder_bbox, height, width):
	mean = np.mean(image)
	std = max(np.std(image), 1e-9)
	new_image = image[bladder_bbox]
	new_image = (new_image - mean) / std
	new_image = resize(new_image, (height, width))
	return new_image

def read_cancer_bbox(filename, image_height, image_width, label):
	with open(filename, 'r') as f:
		cancer_bboxes = pickle.load(f)

	bboxes_image = np.zeros((image_height, image_width))
	grid_x, grid_y = np.mgrid[0:image_height, 0:image_width]
	for box in cancer_bboxes:

		x = box[0]
		y = box[1]
		r = box[2]
		dist_from_center = np.sqrt((grid_x - x) ** 2 + (grid_y - y) ** 2)
		mask = dist_from_center < r
		bboxes_image = np.logical_or(bboxes_image, mask)

	return bboxes_image.astype(np.int)

def shuffle(dataset):
	random.shuffle(dataset)
	return dataset

def next_batch(dataset, batch_size, height, width, epoch):
	images = []
	labels = []

	for i in range(batch_size):

		# dataset[0] = ['D1867766/dwi_ax_0/image_3.npy', 
		#				'D1867766/dwi_ax_0/dilated_mask_3.npy',
		# 				'D1867766/dwi_ax_0/dilated_mask_bbox.json', '0', 
		#				'D1867766/dwi_ax_0/stack0_b0guess/box_label_3.txt']
		ind = batch_size * epoch + i
		image = np.load(os.path.join(senior_path, dataset[ind][0]))
		bladder_bbox = read_bladder_bbox(os.path.join(senior_path, dataset[ind][2]))
		label = dataset[ind][3]

		image_ADC = process_one_channel(image[0], bladder_bbox, height, width)
		image_b0 = process_one_channel(image[1], bladder_bbox, height, width)
		image_b1000 = process_one_channel(image[2], bladder_bbox, height, width)

		processed_image = np.stack([image_ADC, image_b0, image_b1000], axis=2)
		# processed_image.shape = [height, width, 3]
		processed_label = np.zeros((3), np.int32)
		processed_label[int(label)] = 1

        aug_image, _ = augmentation.random_transform(processed_image)


		images.append(aug_image)
		labels.append(processed_label)

	images = np.asarray(images, dtype=np.float32)
	labels = np.asarray(labels)

	return images, labels


def read_and_decode(filename, num_epoch):

	filename_queue = tf.train.string_input_producer([filename], shuffle=False, num_epochs=num_epoch)

	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example, features={
		'label_mul': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.string),
		'image': tf.FixedLenFeature([], tf.string)
		})

	image = tf.decode_raw(features['image'], tf.float32)
	image = tf.reshape(image, [50176,])
	#print(image.shap
	label_mul = tf.decode_raw(features['label_mul'], tf.float32)
	print('label_mul is', type(label_mul))
	label_mul = tf.reshape(label_mul, [4,])
	label = tf.decode_raw(features['label'], tf.float32)
	print('label is', type(label))
	label = tf.reshape(label, [2,])

	#print(type(image))

	return image, label, label_mul

def next_batch(filename, batch_size, num_epoch):

	image, label, label_mul = read_and_decode(filename, num_epoch)
	print("decode OK")
	num_threads = 32
	min_after_dequeue = 10
	capacity = num_threads * batch_size + min_after_dequeue

	image_batch, label_batch, label_mul_batch = tf.train.shuffle_batch(
		[image, label, label_mul], 
		batch_size=batch_size, 
		num_threads=num_threads,
		capacity=capacity, 
		min_after_dequeue=min_after_dequeue)
	print("shuffle OK")

	return  image_batch, label_batch, label_mul_batch






