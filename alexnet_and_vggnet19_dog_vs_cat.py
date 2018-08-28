from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os


class alexnet_and_vggnet19:

	def __init__(self,train_file_path,test_file_path,log_path,model_save_path,net=None,pic_size=None,\
		train_batch_size=None,test_batch_size=None,n_class=None,epoch_num=None):
		self.train_fp = train_file_path  #"/root/damon_files/all/train"
		self.test_fp = test_file_path #"/root/damon_files/all/test1"
		self.lp = log_path
		self.learning_rate = 1e-4
		self.batch_size = train_batch_size
		self.batch_test_size = test_batch_size
		self.n_classes = n_class
		self.n_fc1 = 4096
		self.n_fc2 = 2048
		self.print_step = 10
		self.epoch = epoch_num
		self.save_model_path = model_save_path #"/root/damon_files/model/AlexNet.ckpt"
		self.train_batches_per_epoch = int(np.floor(25000 / self.batch_size))
		self.test_batches_per_epoch = int(np.floor(1000 / self.batch_size))
		self.net_name = net
		self.pic_size = pic_size
		self.get_train_file()
		self.get_train_batch_data(pic_size,pic_size,2048)
		self.get_test_file()
		self.get_test_batch_data(pic_size,pic_size,2048)

	def get_train_file(self):
		'''对数据集文件的位置进行读取，然后根据文件夹名称的不同，将处于不同
			文件夹中的图片标签设置为0或者1，如果有更多分类的话可以依据这个格式
			设置更多的标签类
		'''
		images = []
		temp = []
		for root, sub_folders, files, in os.walk(self.train_fp):
			for name in files:
				images.append(os.path.join(root, name))
			for name in sub_folders:
				temp.append(os.path.join(root, name))
		
		labels = []
		for one_folder in temp[-2:]:
			n_img = len(os.listdir(one_folder))
			letter = one_folder.split('/')[-1]		
			if letter == 'cat':
				labels.extend(n_img*[0])
			else:
				labels.extend(n_img*[1])
				
		temp = np.array([images, labels])
		temp = temp.transpose()
		np.random.shuffle(temp)
		
		self.image_train_list = list(temp[:, 0])
		self.label_train_list = list(temp[:, 1])
		self.label_train_list = [int(float(i)) for i in self.label_train_list]

	def get_train_batch_data(self,img_width,img_height,capacity):
		image = tf.cast(self.image_train_list, tf.string)
		label = tf.cast(self.label_train_list, tf.int32)

		input_queue = tf.train.slice_input_producer([image, label])

		label = input_queue[1]
		image_contents = tf.read_file(input_queue[0])
		image = tf.image.decode_jpeg(image_contents, channels=3)
		image = tf.image.resize_image_with_crop_or_pad(image,img_width,img_height)
		self.image_train_batch, self.label_train_batch = tf.train.batch([image, label], batch_size=self.batch_size, \
													num_threads=64,capacity=capacity)
		self.label_train_batch = tf.reshape(self.label_train_batch, [self.batch_size])

	def get_test_file(self):
		'''对数据集文件的位置进行读取，然后根据文件夹名称的不同，将处于不同
			文件夹中的图片标签设置为0或者1，如果有更多分类的话可以依据这个格式
			设置更多的标签类
		'''
		images = []
		temp = []
		for root, sub_folders, files, in os.walk(self.test_fp):
			for name in files:
				images.append(os.path.join(root, name))
			for name in sub_folders:
				temp.append(os.path.join(root, name))
		
		labels = []
		for one_folder in temp[-2:]:
			n_img = len(os.listdir(one_folder))
			letter = one_folder.split('/')[-1]		
			if letter == 'cat':
				labels.extend(n_img*[0])
			else:
				labels.extend(n_img*[1])
				
		temp = np.array([images, labels])
		temp = temp.transpose()
		np.random.shuffle(temp)
		
		self.image_test_list = list(temp[:, 0])
		self.label_test_list = list(temp[:, 1])
		self.label_test_list = [int(float(i)) for i in self.label_test_list]

	def get_test_batch_data(self,img_width,img_height,capacity):
		image = tf.cast(self.image_test_list, tf.string)
		label = tf.cast(self.label_test_list, tf.int32)

		input_queue = tf.train.slice_input_producer([image, label])

		label = input_queue[1]
		image_contents = tf.read_file(input_queue[0])
		image = tf.image.decode_jpeg(image_contents, channels=3)
		image = tf.image.resize_image_with_crop_or_pad(image,img_width,img_height)
		self.image_test_batch, self.label_test_batch = tf.train.batch([image, label], batch_size=self.batch_test_size, \
													num_threads=64,capacity=capacity)
		self.label_test_batch = tf.reshape(self.label_test_batch, [self.batch_test_size])

	def batch_norm(self,inputs,is_training,is_conv_out=True,decay=0.999):
		scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
		beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
		pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),trainable=False)
		pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

		if is_training:
			if is_conv_out:
				batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
			else:
				batch_mean, batch_var = tf.nn.moments(inputs,[0])

			train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
			train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(inputs,batch_mean,batch_var,beta,scale,0.001)
		else:
			return tf.nn.batch_normalization(inputs,pop_mean,pop_var,beta,scale,0.01)

	def init_param_alexnet(self):
		with tf.name_scope('Inputs'):
			self.x = tf.placeholder(tf.float32, [None, self.pic_size, self.pic_size, 3])
			self.y = tf.placeholder(tf.int32, [None, self.n_classes])
			self.dropout_rate = tf.placeholder(tf.float32)

		with tf.name_scope('Weights'):
			self.W_conv = {
				'conv1' : tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.0001)),
				'conv2' : tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01)),
				'conv3' : tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01)),
				'conv4' : tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01)),
				'conv5' : tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01)),
				'fc6' : tf.Variable(tf.truncated_normal([6 * 6 * 256, self.n_fc1], stddev=0.1)),
				'fc7' : tf.Variable(tf.truncated_normal([self.n_fc1, self.n_fc2], stddev=0.1)),
				'fc8' : tf.Variable(tf.truncated_normal([self.n_fc2, self.n_classes]))
			}
			tf.summary.histogram('weights_conv1',self.W_conv['conv1'])
			tf.summary.histogram('weights_conv2',self.W_conv['conv2'])
			tf.summary.histogram('weights_conv3',self.W_conv['conv3'])
			tf.summary.histogram('weights_conv4',self.W_conv['conv4'])
			tf.summary.histogram('weights_conv5',self.W_conv['conv5'])
			tf.summary.histogram('weights_fc6',self.W_conv['fc6'])
			tf.summary.histogram('weights_fc7',self.W_conv['fc7'])
			tf.summary.histogram('weights_fc8',self.W_conv['fc8'])

		with tf.name_scope('Biases'):
			self.b_conv = {
				'conv1' : tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[96])),
				'conv2' : tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
				'conv3' : tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
				'conv4' : tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
				'conv5' : tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
				'fc6' : tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[self.n_fc1])),
				'fc7' : tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[self.n_fc2])),
				'fc8' : tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.n_classes]))
			}
			tf.summary.histogram('biases_conv1',self.b_conv['conv1'])
			tf.summary.histogram('biases_conv2',self.b_conv['conv2'])
			tf.summary.histogram('biases_conv3',self.b_conv['conv3'])
			tf.summary.histogram('biases_conv4',self.b_conv['conv4'])
			tf.summary.histogram('biases_conv5',self.b_conv['conv5'])
			tf.summary.histogram('biases_fc6',self.b_conv['fc6'])
			tf.summary.histogram('biases_fc7',self.b_conv['fc7'])
			tf.summary.histogram('biases_fc8',self.b_conv['fc8'])

		self.output = self.alexnet()

		with tf.name_scope('Cost'):
			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.y))
			tf.summary.scalar('Cost',self.cost)

		with tf.name_scope('Train'):
			self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

		with tf.name_scope('Prediction'):
			self.correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
			tf.summary.scalar('Accuracy',self.accuracy)

		self.init = tf.global_variables_initializer()

	def alexnet(self):
		with tf.name_scope('alex_net'):
			x_image = tf.reshape(self.x, [-1, self.pic_size, self.pic_size, 3])

			conv1 = tf.nn.conv2d(x_image, self.W_conv['conv1'], strides=[1, 4, 4, 1], padding='VALID')
			conv1 = tf.nn.bias_add(conv1, self.b_conv['conv1'])
			conv1 = self.batch_norm(conv1, True)
			conv1 = tf.nn.relu(conv1)

			# pool1 = tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
			pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
			norm1 = tf.nn.lrn(pool1, 5, bias=1.0, alpha=0.001/9.0, beta=0.75)

			conv2 = tf.nn.conv2d(norm1, self.W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
			conv2 = tf.nn.bias_add(conv2, self.b_conv['conv2'])
			conv2 = self.batch_norm(conv2, True)
			conv2 = tf.nn.relu(conv2)

			# pool2 = tf.nn.avg_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
			pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
			norm2 = tf.nn.lrn(pool2, 5, bias=1.0, alpha=0.001/9.0, beta=0.75)

			conv3 = tf.nn.conv2d(norm2, self.W_conv['conv3'], strides=[1, 1, 1, 1], padding='SAME')
			conv3 = tf.nn.bias_add(conv3, self.b_conv['conv3'])
			conv3 = self.batch_norm(conv3, True)
			conv3 = tf.nn.relu(conv3)

			conv4 = tf.nn.conv2d(conv3, self.W_conv['conv4'], strides=[1, 1, 1, 1], padding='SAME')
			conv4 = tf.nn.bias_add(conv4, self.b_conv['conv4'])
			conv4 = self.batch_norm(conv4, True)
			conv4 = tf.nn.relu(conv4)

			conv5 = tf.nn.conv2d(conv4, self.W_conv['conv5'], strides=[1, 1, 1, 1], padding='SAME')
			conv5 = tf.nn.bias_add(conv5, self.b_conv['conv5'])
			conv5 = self.batch_norm(conv5, True)
			conv5 = tf.nn.relu(conv5)

			# pool5 = tf.nn.avg_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
			pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

			reshape = tf.reshape(pool5, [-1, 6 * 6 * 256])
			fc1 = tf.add(tf.matmul(reshape, self.W_conv['fc6']), self.b_conv['fc6'])
			fc1 = self.batch_norm(fc1, True, False)
			fc1 = tf.nn.relu(fc1)
			fc1 = tf.nn.dropout(fc1, keep_prob=self.dropout_rate)

			fc2 = tf.add(tf.matmul(fc1, self.W_conv['fc7']), self.b_conv['fc7'])
			fc2 = self.batch_norm(fc2, True, False)
			fc2 = tf.nn.relu(fc2)
			fc2 = tf.nn.dropout(fc2, keep_prob=self.dropout_rate)

			fc3 = tf.add(tf.matmul(fc2, self.W_conv['fc8']), self.b_conv['fc8'])

		return fc3

	def init_param_vggnet19(self,npy_path=None):
		if npy_path is not None:
			self.data_dict = np.load(npy_path, encoding='latin1').item()
		else:
			self.data_dict = None

		with tf.name_scope("Inputs"):
			self.x = tf.placeholder(tf.float32, [None,self.pic_size,self.pic_size,3])
			self.y = tf.placeholder(tf.float32, [None, self.n_classes])
			self.dropout_rate = tf.placeholder(tf.float32)

		with tf.name_scope("Weights"):
			if self.data_dict == None:
				self.W_conv = {
					'conv1_1' : tf.Variable(tf.truncated_normal([3, 3, 3, 64], 0.0, 0.001)),
					'conv1_2' : tf.Variable(tf.truncated_normal([3, 3, 64, 64], 0.0, 0.001)),
					'conv2_1' : tf.Variable(tf.truncated_normal([3, 3, 64, 128], 0.0, 0.001)),
					'conv2_2' : tf.Variable(tf.truncated_normal([3, 3, 128, 128], 0.0, 0.001)),
					'conv3_1' : tf.Variable(tf.truncated_normal([3, 3, 128, 256], 0.0, 0.001)),
					'conv3_2' : tf.Variable(tf.truncated_normal([3, 3, 256, 256], 0.0, 0.001)),
					'conv3_3' : tf.Variable(tf.truncated_normal([3, 3, 256, 256], 0.0, 0.001)),
					'conv3_4' : tf.Variable(tf.truncated_normal([3, 3, 256, 256], 0.0, 0.001)),
					'conv4_1' : tf.Variable(tf.truncated_normal([3, 3, 256, 512], 0.0, 0.001)),
					'conv4_2' : tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0.0, 0.001)),
					'conv4_3' : tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0.0, 0.001)),
					'conv4_4' : tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0.0, 0.001)),
					'conv5_1' : tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0.0, 0.001)),
					'conv5_2' : tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0.0, 0.001)),
					'conv5_3' : tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0.0, 0.001)),
					'conv5_4' : tf.Variable(tf.truncated_normal([3, 3, 512, 512], 0.0, 0.001)),
					'fc6' : tf.Variable(tf.truncated_normal([7 * 7 * 512, self.n_fc1], 0.0, 0.001)),
					'fc7' : tf.Variable(tf.truncated_normal([self.n_fc1, self.n_fc2], 0.0, 0.001)),
					'fc8' : tf.Variable(tf.truncated_normal([self.n_fc2, 1000], 0.0, 0.001)),
					'fc9' : tf.Variable(tf.truncated_normal([1000, self.n_classes], 0.0, 0.001))
				}
			else:
				self.W_conv = {}
				for name in list(self.data_dict.keys()):
					self.W_conv.update({name : tf.Variable(self.data_dict[name][0])})
				self.W_conv.update({'fc9' : tf.Variable(tf.truncated_normal([1000, self.n_classes], 0.0, 0.001))})

			for key, value in self.W_conv.items():
				tf.summary.histogram('weights_' + key, value)

		with tf.name_scope("Biases"):
			if self.data_dict == None:
				self.b_conv = {
					'conv1' : tf.Variable(tf.truncated_normal([64], 0.0, 0.001)),
					'conv1_1' : tf.Variable(tf.truncated_normal([64], 0.0, 0.001)),
					'conv1_2' : tf.Variable(tf.truncated_normal([64], 0.0, 0.001)),
					'conv2_1' : tf.Variable(tf.truncated_normal([128], 0.0, 0.001)),
					'conv2_2' : tf.Variable(tf.truncated_normal([128], 0.0, 0.001)),
					'conv3_1' : tf.Variable(tf.truncated_normal([256], 0.0, 0.001)),
					'conv3_2' : tf.Variable(tf.truncated_normal([256], 0.0, 0.001)),
					'conv3_3' : tf.Variable(tf.truncated_normal([256], 0.0, 0.001)),
					'conv3_4' : tf.Variable(tf.truncated_normal([256], 0.0, 0.001)),
					'conv4_1' : tf.Variable(tf.truncated_normal([512], 0.0, 0.001)),
					'conv4_2' : tf.Variable(tf.truncated_normal([512], 0.0, 0.001)),
					'conv4_3' : tf.Variable(tf.truncated_normal([512], 0.0, 0.001)),
					'conv4_4' : tf.Variable(tf.truncated_normal([512], 0.0, 0.001)),
					'conv5_1' : tf.Variable(tf.truncated_normal([512], 0.0, 0.001)),
					'conv5_2' : tf.Variable(tf.truncated_normal([512], 0.0, 0.001)),
					'conv5_3' : tf.Variable(tf.truncated_normal([512], 0.0, 0.001)),
					'conv5_4' : tf.Variable(tf.truncated_normal([512], 0.0, 0.001)),
					'fc6' : tf.Variable(tf.truncated_normal([self.n_fc1], 0.0, 0.001)),
					'fc7' : tf.Variable(tf.truncated_normal([self.n_fc2], 0.0, 0.001)),
					'fc8' : tf.Variable(tf.truncated_normal([1000], 0.0, 0.001)),
					'fc9' : tf.Variable(tf.truncated_normal([self.n_classes], 0.0, 0.001))
				}
			else: 
				self.b_conv = {}
				for name in list(self.data_dict.keys()):
					self.b_conv.update({name : tf.Variable(self.data_dict[name][1])})
				self.b_conv.update({'fc9' : tf.Variable(tf.truncated_normal([self.n_classes], 0.0, 0.001))})

			for key, value in self.b_conv.items():
				tf.summary.histogram('biases_' + key, value)

		self.output = self.vggnet19()

		with tf.name_scope('Cost'):
			self.cost = tf.reduce_sum((self.output - self.y) ** 2)
			tf.summary.scalar('Cost', self.cost)

		with tf.name_scope('Train'):
			self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

		with tf.name_scope('Prediction'):
			self.correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
			tf.summary.scalar('Accuracy', self.accuracy)

		self.init = tf.global_variables_initializer()

	def vggnet19(self):
		with tf.name_scope("vgg_net_19"):
			x_image = tf.reshape(self.x, [-1, self.pic_size, self.pic_size, 3])

			# conv1_1
			conv1_1 = tf.nn.conv2d(x_image, self.W_conv['conv1_1'], strides=[1,1,1,1], padding='SAME')
			conv1_1 = tf.nn.bias_add(conv1_1, self.b_conv['conv1_1'])
			conv1_1 = tf.nn.relu(conv1_1)
			# conv1_2
			conv1_2 = tf.nn.conv2d(conv1_1, self.W_conv['conv1_2'], strides=[1,1,1,1], padding='SAME')
			conv1_2 = tf.nn.bias_add(conv1_2, self.b_conv['conv1_2'])
			conv1_2 = tf.nn.relu(conv1_2)

			pool_1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

			# conv2_1
			conv2_1 = tf.nn.conv2d(pool_1, self.W_conv['conv2_1'], strides=[1,1,1,1], padding='SAME')
			conv2_1 = tf.nn.bias_add(conv2_1, self.b_conv['conv2_1'])
			conv2_1 = tf.nn.relu(conv2_1)
			# conv2_2
			conv2_2 = tf.nn.conv2d(conv2_1, self.W_conv['conv2_2'], strides=[1,1,1,1], padding='SAME')
			conv2_2 = tf.nn.bias_add(conv2_2, self.b_conv['conv2_2'])
			conv2_2 = tf.nn.relu(conv2_2)

			pool_2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

			# conv3_1
			conv3_1 = tf.nn.conv2d(pool_2, self.W_conv['conv3_1'], strides=[1,1,1,1], padding='SAME')
			conv3_1 = tf.nn.bias_add(conv3_1, self.b_conv['conv3_1'])
			conv3_1 = tf.nn.relu(conv3_1)
			# conv3_2
			conv3_2 = tf.nn.conv2d(conv3_1, self.W_conv['conv3_2'], strides=[1,1,1,1], padding='SAME')
			conv3_2 = tf.nn.bias_add(conv3_2, self.b_conv['conv3_2'])
			conv3_2 = tf.nn.relu(conv3_2)
			# conv3_3
			conv3_3 = tf.nn.conv2d(conv3_2, self.W_conv['conv3_3'], strides=[1,1,1,1], padding='SAME')
			conv3_3 = tf.nn.bias_add(conv3_3, self.b_conv['conv3_3'])
			conv3_3 = tf.nn.relu(conv3_3)
			# conv3_4 
			conv3_4 = tf.nn.conv2d(conv3_3, self.W_conv['conv3_4'], strides=[1,1,1,1], padding='SAME')
			conv3_4 = tf.nn.bias_add(conv3_4, self.b_conv['conv3_4'])
			conv3_4 = tf.nn.relu(conv3_4)

			pool_3 = tf.nn.max_pool(conv3_4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

			# conv4_1
			conv4_1 = tf.nn.conv2d(pool_3, self.W_conv['conv4_1'], strides=[1,1,1,1], padding='SAME')
			conv4_1 = tf.nn.bias_add(conv4_1, self.b_conv['conv4_1'])
			conv4_1 = tf.nn.relu(conv4_1)
			# conv4_2
			conv4_2 = tf.nn.conv2d(conv4_1, self.W_conv['conv4_2'], strides=[1,1,1,1], padding='SAME')
			conv4_2 = tf.nn.bias_add(conv4_2, self.b_conv['conv4_2'])
			conv4_2 = tf.nn.relu(conv4_2)
			# conv4_3
			conv4_3 = tf.nn.conv2d(conv4_2, self.W_conv['conv4_3'], strides=[1,1,1,1], padding='SAME')
			conv4_3 = tf.nn.bias_add(conv4_3, self.b_conv['conv4_3'])
			conv4_3 = tf.nn.relu(conv4_3)
			# conv4_4 
			conv4_4 = tf.nn.conv2d(conv4_3, self.W_conv['conv4_4'], strides=[1,1,1,1], padding='SAME')
			conv4_4 = tf.nn.bias_add(conv4_4, self.b_conv['conv4_4'])
			conv4_4 = tf.nn.relu(conv4_4)

			pool_4 = tf.nn.max_pool(conv4_4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

			# conv5_1
			conv5_1 = tf.nn.conv2d(pool_4, self.W_conv['conv5_1'], strides=[1,1,1,1], padding='SAME')
			conv5_1 = tf.nn.bias_add(conv5_1, self.b_conv['conv5_1'])
			conv5_1 = tf.nn.relu(conv5_1)
			# conv5_2
			conv5_2 = tf.nn.conv2d(conv5_1, self.W_conv['conv5_2'], strides=[1,1,1,1], padding='SAME')
			conv5_2 = tf.nn.bias_add(conv5_2, self.b_conv['conv5_2'])
			conv5_2 = tf.nn.relu(conv5_2)
			# conv5_3
			conv5_3 = tf.nn.conv2d(conv5_2, self.W_conv['conv5_3'], strides=[1,1,1,1], padding='SAME')
			conv5_3 = tf.nn.bias_add(conv5_3, self.b_conv['conv5_3'])
			conv5_3 = tf.nn.relu(conv5_3)
			# conv5_4 
			conv5_4 = tf.nn.conv2d(conv5_3, self.W_conv['conv5_4'], strides=[1,1,1,1], padding='SAME')
			conv5_4 = tf.nn.bias_add(conv5_4, self.b_conv['conv5_4'])
			conv5_4 = tf.nn.relu(conv5_4)

			pool_5 = tf.nn.max_pool(conv5_4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

			reshape = tf.reshape(pool_5, [-1, 7 * 7 * 512])

			fc6 = tf.nn.bias_add(tf.matmul(reshape, self.W_conv['fc6']), self.b_conv['fc6'])
			fc6 = tf.nn.relu(fc6)
			fc6 = tf.nn.dropout(fc6, keep_prob=self.dropout_rate)

			fc7 = tf.nn.bias_add(tf.matmul(fc6, self.W_conv['fc7']), self.b_conv['fc7'])
			fc7 = tf.nn.relu(fc7)
			fc7 = tf.nn.dropout(fc7, keep_prob=self.dropout_rate)

			fc8 = tf.nn.bias_add(tf.matmul(fc7, self.W_conv['fc8']), self.b_conv['fc8'])
			fc8 = tf.nn.relu(fc8)
			fc8 = tf.nn.dropout(fc8, keep_prob=self.dropout_rate)

			fc9 = tf.nn.bias_add(tf.matmul(fc8, self.W_conv['fc9']), self.b_conv['fc9'])

		return tf.nn.softmax(fc9)

	def onehot(self,labels):
		n_sample = len(labels)
		n_class = max(labels) + 1
		onehot_labels = np.zeros((n_sample, n_class))
		onehot_labels[np.arange(n_sample), labels] = 1
		return onehot_labels

	def run(self):
		if self.net_name == 'alexnet':
			self.init_param_alexnet()
			print(" --------------------- Start Training AlexNet ---------------------")
		elif self.net_name == 'vggnet19':
			self.init_param_vggnet19(npy_path='./vgg19.npy')
			print(" --------------------- Start Training VGGNet19 ---------------------")

		with tf.Session() as sess:
			sess.run(self.init)
			merged = tf.summary.merge_all()
			writer = tf.summary.FileWriter(self.lp, sess.graph)
			saver = tf.train.Saver()

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			for i in range(self.epoch):
				step = 1
				train_acc = 0.0
				train_loss = 0.0
				train_count = 0
				for step in range(self.train_batches_per_epoch):
					image_train, label_train = sess.run([self.image_train_batch, self.label_train_batch])
					label_train = self.onehot(label_train)
					if label_train.shape[1] == self.n_classes:
						sess.run(self.optimizer, feed_dict={self.x:image_train, self.y:label_train, self.dropout_rate:0.5})
						if step % self.print_step == 0:
							acc, loss, summary = sess.run([self.accuracy, self.cost, merged], feed_dict={self.x:image_train, 
														self.y:label_train, self.dropout_rate:1.0})
							train_acc += acc
							train_loss += loss
							train_count += 1
							writer.add_summary(summary, i * self.train_batches_per_epoch + step)
				print('[*] Epoch ' + str(i+1) + ': avg_train_acc = ', np.around(train_acc / train_count, 4), 
						', avg_loss = ', np.around(train_loss / train_count, 4))
				print(" -----------------------------------------------------------------")

			print("[*] Optimization Finished!")
			saver.save(sess, self.save_model_path)
			print("[*] Model Save Finished!")
			coord.request_stop()
			coord.join(threads)			

			test_acc = 0.0
			test_loss = 0.0
			test_count = 0
			for _ in range(self.test_batches_per_epoch):
				image_test, label_test = sess.run([self.image_test_batch, self.label_test_batch])
				label_test = self.onehot(label_test)
				acc = sess.run(self.accuracy, feed_dict={self.x:image_test, self.y:label_test, self.dropout_rate:1.0})
				test_acc += acc
				test_count += 1
			test_acc /= test_count

			print(" [*] Validation: test dataset acc = " + str(np.around(test_acc, 4)))


if __name__ == '__main__':
	train_data_path = "/root/damon_files/all/train"
	test_data_path = "/root/damon_files/all/test2"
	log_path = "/root/damon_files/log" 
	model_path = "/root/damon_files/model"
	# dog_vs_cat = alexnet_and_vggnet19(train_data_path,test_data_path,log_path,model_path,net='alexnet',pic_size=227,\
	# 							train_batch_size=200, test_batch_size=200, n_class=2, epoch_num=300)
	dog_vs_cat = alexnet_and_vggnet19(train_data_path,test_data_path,log_path,model_path,net='vggnet19',pic_size=224,\
								train_batch_size=10, test_batch_size=10, n_class=2, epoch_num=30)
	dog_vs_cat.run()
