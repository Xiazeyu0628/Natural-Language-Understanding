# coding: utf-8
import sys
import time
import numpy as np

from utils import *
from rnnmath import *
from sys import stdout




class RNN(object):
	'''
	This class implements Recurrent Neural Networks.
	
	You should implement code in the following functions:
		predict				->	predict an output sequence for a given input sequence
		acc_deltas			->	accumulate update weights for the RNNs weight matrices, standard Back Propagation
		acc_deltas_bptt		->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
		acc_deltas_np		->	accumulate update weights for the RNNs weight matrices, standard Back Propagation -- for number predictions
		acc_deltas_bptt_np	->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time -- for number predictions
		compute_loss 		->	compute the (cross entropy) loss between the desired output and predicted output for a given input sequence
		compute_mean_loss	->	compute the average loss over all sequences in a corpus
		generate_sequence	->	use the RNN to generate a new (unseen) sequnce
	
	Do NOT modify any other methods!
	Do NOT change any method signatures!
	'''
	
	def __init__(self, vocab_size, hidden_dims, out_vocab_size):
		'''
		initialize the RNN with random weight matrices.
		
		DO NOT CHANGE THIS
		
		vocab_size		size of vocabulary that is being used
		hidden_dims		number of hidden units
		out_vocab_size	size of the output vocabulary
		'''
		self.vocab_size = vocab_size
		self.hidden_dims = hidden_dims
		self.out_vocab_size = out_vocab_size 
		
		# matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
		self.U = np.random.randn(self.hidden_dims, self.hidden_dims)*np.sqrt(0.1)
		self.V = np.random.randn(self.hidden_dims, self.vocab_size)*np.sqrt(0.1)
		self.W = np.random.randn(self.out_vocab_size, self.hidden_dims)*np.sqrt(0.1)
		
		# matrices to accumulate weight updates
		self.deltaU = np.zeros((self.hidden_dims, self.hidden_dims))
		self.deltaV = np.zeros((self.hidden_dims, self.vocab_size))
		self.deltaW = np.zeros((self.out_vocab_size, self.hidden_dims))

	def apply_deltas(self, learning_rate):
		'''
		update the RNN's weight matrices with corrections accumulated over some training instances
		
		DO NOT CHANGE THIS
		
		learning_rate	scaling factor for update weights
		'''
		# apply updates to U, V, W
		self.U += learning_rate*self.deltaU
		self.W += learning_rate*self.deltaW
		self.V += learning_rate*self.deltaV
		
		# reset matrices
		self.deltaU.fill(0.0)
		self.deltaV.fill(0.0)
		self.deltaW.fill(0.0)
	
	def predict(self, x):
		'''
		predict an output sequence y for a given input sequence x
		
		x	list of words, as indices, e.g.: [0, 4, 2]   每次输入一个word,每次下个word的概率都会有改变
		
		returns	y,s
		y	matrix of probability vectors for each input word
		s	matrix of hidden layers for each input word
		
		'''
		
		# matrix s for hidden states, y for output states, given input x.
		# rows correspond to times t, i.e., input words
		# s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )
		s = np.zeros((len(x) + 1, self.hidden_dims))   # 第一行为全0，从第二行开始输入
		y = np.zeros((len(x), self.out_vocab_size))
		
		for t in range(len(x)):  # t=0 仍然代表的是第一个时间点， 现在s[0]和y[0] 都代表第一个时间的值，之前这并不统一。
			##########################
			# --- your code here --- #
			word_index = x[t]
			V_x = self.V[:, word_index]  # V_x size self.hidden_dims*1
			U_s = np.dot(self.U, s[t-1, :].T)  # 这里用的s是上一个时刻的
			net_in = V_x + U_s
			#St = 1 / (1 + np.exp(-net_in))
			St = sigmoid(net_in)
			s[[t], :] = St
			net_out = np.dot(self.W, s[t, :].T)  # net_out size: self.out_vocab_size*1
			#y[[t + 1], :] = net_out.T / sum(net_out)
			y[[t], :] = softmax(net_out)
			##########################
		
		return y, s
	
	def acc_deltas(self, x, d, y, s):
		'''
		accumulate updates for V, W, U   变化的量，这里没有用learning rate
		standard back propagation
		
		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
		
		x	list of words, as indices, e.g.: [0, 4, 2]
		d	list of words, as indices, e.g.: [4, 2, 3]
		y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
			should be part of the return value of predict(x)
		s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
			should be part of the return value of predict(x)
		
		no return values
		'''
		for t in reversed(range(len(x))):
			##########################
			# --- your code here --- #
			y_t = y[t, :]
			one_hot_index_d = d[t]  # 只有在此index处的值为1 其他地方为0
			#d_t = np.zeros(1, self.vocab_size)
			#d_t[0, one_hot_index_d] = 1
			d_t = make_onehot(one_hot_index_d,self.vocab_size)
			derivative_softmax = np.ones(self.vocab_size)
			delta_out = (d_t - y_t) * derivative_softmax  # element-wise product 1*vocab_size
			s_t = s[t, :]  # 1*self.hidden_dims
			deltaW = np.outer(delta_out, s_t)  # 向量外积       2000*hidden_dims

			# ----#

			derivative_sigmoid = s_t * (np.ones(self.hidden_dims) - s_t)    # element-wise product  1*vocab_size

			delta_in = np.dot(self.W.T, delta_out) * derivative_sigmoid
			one_hot_index_x = x[t]
			x_t = make_onehot(one_hot_index_x,self.vocab_size)
			deltaV = np.outer(delta_in, x_t)  # 向量外积
			deltaU = np.outer(delta_in, s[t - 1, :])

			self.deltaV += deltaV
			self.deltaU += deltaU
			self.deltaW += deltaW



		##########################

	def acc_deltas_np(self, x, d, y, s):
		'''
		accumulate updates for V, W, U
		standard back propagation
		
		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
		for number prediction task, we do binary prediction, 0 or 1

		x	list of words, as indices, e.g.: [0, 4, 2]
		d	array with one element, as indices, e.g.: [0] or [1]
		y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
			should be part of the return value of predict(x)
		s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
			should be part of the return value of predict(x)
		
		no return values
		'''
		
		##########################
		# --- your code here --- #
		##########################
		
		
	def acc_deltas_bptt(self, x, d, y, s, steps):
		'''
		accumulate updates for V, W, U
		back propagation through time (BPTT)
		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
		x		list of words, as indices, e.g.: [0, 4, 2]
		d		list of words, as indices, e.g.: [4, 2, 3]
		y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
				should be part of the return value of predict(x)
		s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
				should be part of the return value of predict(x)
		steps	number of time steps to go back in BPTT
		no return values
		'''

		for t in reversed(range(len(x))):
			# t from 0 to len(x)-1
			# print("time {0}".format(t))
			##########################
			# --- your code here --- #
			# deltaW is update as before
			y_t = y[t, :]
			one_hot_index = d[t]  # 只有在此index处的值为1 其他地方为0
			d_t = make_onehot(one_hot_index,self.vocab_size)

			derivative_softmax = np.ones(self.vocab_size)
			delta_out = (d_t - y_t) * derivative_softmax
			s_t = s[t, :]
			deltaW = np.outer(delta_out, s_t)  # 向量外积
			self.deltaW += deltaW

			# 在本测试中，steps = 3  	t = range(1,7)
			  # 这个条件允许通过的t =  3 4 5 6       该条件来源于在更新deltaU时要用到S（t-steps-1）,而s的最小index为-1 代表初始时刻的s值
			deltaU = np.zeros((self.hidden_dims, self.hidden_dims))
			deltaV = np.zeros((self.hidden_dims, self.vocab_size))
			for i in range(0,steps+1):
				if (t - i - 1 < -1):
					break
				if (t - i - 1 >= -1):
					s_i = s[(t-i),:]   #t时刻的s为s[t],t-i 时刻的就为s[t-i]
					derivative_sigmoid_i=  s_i * (np.ones(self.hidden_dims) - s_i)
					if i == 0:  # i == 0 时， 采用 如1.c 里的计算方法 算出 delta_in（t） 的值
						delta_in_t = np.dot(self.W.T, delta_out) * derivative_sigmoid_i
						delta_in_last = delta_in_t  #
					if i != 0:  # i ！= 时 迭代求得 delta_in（t-steps），为了能取到steps，所以range函数里的steps+1
						delta_in_i = np.dot(self.U.T, delta_in_last) * derivative_sigmoid_i    # 左边为上一个时刻的delta_in 等式右边为写一个时刻的delta_in. 例如 delta_in（t-1） 和  delta_in（t）
						delta_in_last = delta_in_i
				one_hot_index_x = x[t-i]
				x_t = make_onehot(one_hot_index_x,self.vocab_size)
				deltaV += np.outer(delta_in_last,x_t)
				deltaU += np.outer(delta_in_last,s[(t-i-1),:])
			self.deltaU += deltaU
			self.deltaV += deltaV




			##########################

	def acc_deltas_bptt_np(self, x, d, y, s, steps):
		'''
		accumulate updates for V, W, U
		back propagation through time (BPTT)
		
		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
		for number prediction task, we do binary prediction, 0 or 1

		x	list of words, as indices, e.g.: [0, 4, 2]
		d	array with one element, as indices, e.g.: [0] or [1]
		y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
				should be part of the return value of predict(x)
		s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
				should be part of the return value of predict(x)
		steps	number of time steps to go back in BPTT
		
		no return values
		'''
		
		##########################
		# --- your code here --- #
		##########################


	def compute_loss(self, x, d):
		'''
		compute the loss between predictions y for x, and desired output d.
		
		first predicts the output for x using the RNN, then computes the loss w.r.t. d
		
		x		list of words, as indices, e.g.: [0, 4, 2]
		d		list of words, as indices, e.g.: [4, 2, 3]
		
		return loss		the combined loss for all words
		'''
		
		loss = 0.

		##########################
		# --- your code here --- #
		loss_matrix = np.zeros((1,len(x)))
		y,s = self.predict(x)    # 得到基于 x的预测矩阵
		for time_step in range(len(x)):
			word_index = d[time_step]   # 对于 one-hot vector of word in d 只有在word_index处的值为1 ，其他地方的值为0
			y_t = y[time_step,:]   # predict probability of vocabulary at time = time_step
			jj = -1*np.log(y_t[word_index])
			loss_matrix[0,time_step] = jj
		loss = np.sum(loss_matrix)
		##########################
		return loss


	def compute_loss_np(self, x, d):
		'''
		compute the loss between predictions y for x, and desired output d.
		
		first predicts the output for x using the RNN, then computes the loss w.r.t. d
		
		x		list of words, as indices, e.g.: [0, 4, 2]
		d		a word, as indices, e.g.: [0]
		
		return loss		we only take the prediction from the last time step
		'''
		
		loss = 0.
		
		##########################
		# --- your code here --- #
		##########################
		
		return loss


	def compute_acc_np(self, x, d):
		'''
		compute the accuracy prediction, y[t] compared to the desired output d.
		first predicts the output for x using the RNN, then computes the loss w.r.t. d
		
		x		list of words, as indices, e.g.: [0, 4, 2]
		d		a word class (plural/singular), as index, e.g.: [0] or [1]
		
		return 1 if argmax(y[t]) == d[0], 0 otherwise
		'''
		

		##########################
		# --- your code here --- #
		##########################
		
		return 0


	def compare_num_pred(self, x, d):
		'''
		compute the probability between predictions the desired output d[0] and it's (re)inflected form, d[1].
		first predicts the output for x using the RNN, then compare the probability of d[0] and d[1].
		
		x		list of words, as indices, e.g.: [0, 4, 2]
		d		the desired verb and its (re)inflected form (singular/plural), as indices, e.g.: [7, 8]
		
		return 1 if p(d[0]) > p(d[1]), 0 otherwise
		'''
		
		##########################
		# --- your code here --- #
		##########################
		
		return 0


	def compute_acc_lmnp(self, X_dev, D_dev):
		'''
		
		DO NOT CHANGE THIS
		
		X_dev			a list of input vectors, e.g., 		[[5, 4, 2], [7, 3, 8]]
		D_dev			a list of pair verb forms (plural/singular), e.g., 	[[4, 9], [6, 5]]
		'''
		acc = sum([self.compare_num_pred(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / len(X_dev)

		return acc

		
	def compute_mean_loss(self, X, D):
		'''
		compute the mean loss between predictions for corpus X and desired outputs in corpus D.
		
		X		corpus of sentences x1, x2, x3, [...], each a list of words as indices.
		D		corpus of desired outputs d1, d2, d3 [...], each a list of words as indices.
		
		return mean_loss		average loss over all words in D
		'''
		
		mean_loss = 0.
		total_word_number = 0
		total_loss = 0
		##########################
		sentence_number = len(X)   # 句子的数量
		for i in range(sentence_number):
			sentence = X[i]
			word_number = sentence.shape[0]  # 每一行句子的词的个数
			sum_loss = self.compute_loss(X[i],D[i])
			total_word_number = total_word_number + word_number
			total_loss = total_loss + sum_loss
		mean_loss = total_loss/total_word_number
		##########################
		
		return mean_loss
	
		
	def train(self, X, D, X_dev, D_dev, epochs=10, learning_rate=0.5, anneal=5, back_steps=0, batch_size=100, min_change=0.0001, log=True):
		'''
		train the RNN on some training set X, D while optimizing the loss on a dev set X_dev, D_dev
		
		DO NOT CHANGE THIS
		
		training stops after the first of the following is true:
			* number of epochs reached
			* minimum change observed for more than 2 consecutive epochs
		
		X				a list of input vectors, e.g., 		[[0, 4, 2], [1, 3, 0]]
		D				a list of desired outputs, e.g., 	[[4, 2, 3], [3, 0, 3]]
		X_dev			a list of input vectors, e.g., 		[[0, 4, 2], [1, 3, 0]]
		D_dev			a list of desired outputs, e.g., 	[[4, 2, 3], [3, 0, 3]]
		epochs			maximum number of epochs (iterations) over the training set. default 10
		learning_rate	initial learning rate for training. default 0.5
		anneal			positive integer. if > 0, lowers the learning rate in a harmonically after each epoch.
						higher annealing rate means less change per epoch.
						anneal=0 will not change the learning rate over time.
						default 5
		back_steps		positive integer. number of timesteps for BPTT. if back_steps < 2, standard BP will be used. default 0
		batch_size		number of training instances to use before updating the RNN's weight matrices.
						if set to 1, weights will be updated after each instance. if set to len(X), weights are only updated after each epoch.
						default 100
		min_change		minimum change in loss between 2 epochs. if the change in loss is smaller than min_change, training stops regardless of
						number of epochs left.
						default 0.0001
		log				whether or not to print out log messages. (default log=True)
		'''
		if log:
			stdout.write("\nTraining model for {0} epochs\ntraining set: {1} sentences (batch size {2})".format(epochs, len(X), batch_size))
			stdout.write("\nOptimizing loss on {0} sentences".format(len(X_dev)))
			stdout.write("\nVocab size: {0}\nHidden units: {1}".format(self.vocab_size, self.hidden_dims))
			stdout.write("\nSteps for back propagation: {0}".format(back_steps))
			stdout.write("\nInitial learning rate set to {0}, annealing set to {1}".format(learning_rate, anneal))
			stdout.write("\n\ncalculating initial mean loss on dev set")
			stdout.flush()
		
		t_start = time.time()
		loss_function = self.compute_loss
		
		loss_sum = sum([len(d) for d in D_dev])
		initial_loss = sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / loss_sum

		if log or not log:
			stdout.write(": {0}\n".format(initial_loss))
			stdout.flush()
		
		prev_loss = initial_loss
		loss_watch_count = -1
		min_change_count = -1
		
		a0 = learning_rate
		
		best_loss = initial_loss
		bestU, bestV, bestW = self.U, self.V, self.W
		best_epoch = 0
		
		for epoch in range(epochs):
			if anneal > 0:
				learning_rate = a0/((epoch+0.0+anneal)/anneal)
			else:
				learning_rate = a0
			
			if log:
				stdout.write("\nepoch %d, learning rate %.04f" % (epoch+1, learning_rate))
				stdout.flush()
			
			t0 = time.time()
			count = 0
			
			# use random sequence of instances in the training set (tries to avoid local maxima when training on batches)
			permutation = np.random.permutation(range(len(X)))
			if log:
				stdout.write("\tinstance 1")
			for i in range(len(X)):
				c = i+1
				if log:
					stdout.write("\b"*len(str(i)))
					stdout.write("{0}".format(c))
					stdout.flush()
				p = permutation[i]
				x_p = X[p]
				d_p = D[p]
				
				y_p, s_p = self.predict(x_p)
				if back_steps == 0:
					self.acc_deltas(x_p, d_p, y_p, s_p)
				else:
					self.acc_deltas_bptt(x_p, d_p, y_p, s_p, back_steps)
				
				if i % batch_size == 0:
					self.deltaU /= batch_size
					self.deltaV /= batch_size
					self.deltaW /= batch_size
					self.apply_deltas(learning_rate)
			
			if len(X) % batch_size > 0:
				mod = len(X) % batch_size
				self.deltaU /= mod
				self.deltaV /= mod
				self.deltaW /= mod
				self.apply_deltas(learning_rate)
			
			loss = sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))])/loss_sum
			
			if log:
				stdout.write("\tepoch done in %.02f seconds" % (time.time() - t0))
				stdout.write("\tnew loss: {0}".format(loss))
				stdout.flush()
			
			if loss < best_loss:
				best_loss = loss
				bestU, bestV, bestW = self.U.copy(), self.V.copy(), self.W.copy()
				best_epoch = epoch
			
			# make sure we change the RNN enough
			if abs(prev_loss - loss) < min_change:
				min_change_count += 1
			else:
				min_change_count = 0
			if min_change_count > 2:
				print("\n\ntraining finished after {0} epochs due to minimal change in loss".format(epoch+1))
				break
			
			prev_loss = loss
		
		t = time.time() - t_start
		
		if min_change_count <= 2:
			print("\n\ntraining finished after reaching maximum of {0} epochs".format(epochs))
		print("best observed loss was {0}, at epoch {1}".format(best_loss, (best_epoch+1)))
		
		print("setting U, V, W to matrices from best epoch")
		self.U, self.V, self.W = bestU, bestV, bestW
		
		return best_loss


	def train_np(self, X, D, X_dev, D_dev, epochs=10, learning_rate=0.5, anneal=5, back_steps=0, batch_size=100, min_change=0.0001, log=True):
		'''
		train the RNN on some training set X, D while optimizing the loss on a dev set X_dev, D_dev

		DO NOT CHANGE THIS
		
		training stops after the first of the following is true:
			* number of epochs reached
			* minimum change observed for more than 2 consecutive epochs
		
		X				a list of input vectors, e.g., 		[[5, 4, 2], [7, 3, 8]]
		D				a list of desired outputs, e.g., 	[[0], [1]]
		X_dev			a list of input vectors, e.g., 		[[5, 4, 2], [7, 3, 8]]
		D_dev			a list of desired outputs, e.g., 	[[0], [1]]
		epochs			maximum number of epochs (iterations) over the training set. default 10
		learning_rate	initial learning rate for training. default 0.5
		anneal			positive integer. if > 0, lowers the learning rate in a harmonically after each epoch.
						higher annealing rate means less change per epoch.
						anneal=0 will not change the learning rate over time.
						default 5
		back_steps		positive integer. number of timesteps for BPTT. if back_steps < 2, standard BP will be used. default 0
		batch_size		number of training instances to use before updating the RNN's weight matrices.
						if set to 1, weights will be updated after each instance. if set to len(X), weights are only updated after each epoch.
						default 100
		min_change		minimum change in loss between 2 epochs. if the change in loss is smaller than min_change, training stops regardless of
						number of epochs left.
						default 0.0001
		log				whether or not to print out log messages. (default log=True)
		'''
		if log:
			stdout.write("\nTraining model for {0} epochs\ntraining set: {1} sentences (batch size {2})".format(epochs, len(X), batch_size))
			stdout.write("\nOptimizing loss on {0} sentences".format(len(X_dev)))
			stdout.write("\nVocab size: {0}\nHidden units: {1}".format(self.vocab_size, self.hidden_dims))
			stdout.write("\nSteps for back propagation: {0}".format(back_steps))
			stdout.write("\nInitial learning rate set to {0}, annealing set to {1}".format(learning_rate, anneal))
			stdout.flush()
		
		t_start = time.time()
		loss_function = self.compute_loss_np
		loss_sum = len(D_dev)
		initial_loss = sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / loss_sum
		initial_acc = sum([self.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / len(X_dev)

		if log or not log:
			stdout.write("\n\ncalculating initial mean loss on dev set")
			stdout.write(": {0}\n".format(initial_loss))
			stdout.write("calculating initial acc on dev set")
			stdout.write(": {0}\n".format(initial_acc))
			stdout.flush()
		
		prev_loss = initial_loss
		loss_watch_count = -1
		min_change_count = -1
		
		a0 = learning_rate
		
		best_loss = initial_loss
		bestU, bestV, bestW = self.U, self.V, self.W
		best_epoch = 0
		
		for epoch in range(epochs):
			if anneal > 0:
				learning_rate = a0/((epoch+0.0+anneal)/anneal)
			else:
				learning_rate = a0
			
			if log:
				stdout.write("\nepoch %d, learning rate %.04f" % (epoch+1, learning_rate))
				stdout.flush()
			
			t0 = time.time()
			count = 0
			
			# use random sequence of instances in the training set (tries to avoid local maxima when training on batches)
			permutation = np.random.permutation(range(len(X)))
			if log:
				stdout.write("\tinstance 1")
			for i in range(len(X)):
				c = i+1
				if log:
					stdout.write("\b"*len(str(i)))
					stdout.write("{0}".format(c))
					stdout.flush()
				p = permutation[i]
				x_p = X[p]
				d_p = D[p]
				
				y_p, s_p = self.predict(x_p)
				if back_steps == 0:
					self.acc_deltas_np(x_p, d_p, y_p, s_p)
				else:
					self.acc_deltas_bptt_np(x_p, d_p, y_p, s_p, back_steps)
				
				if i % batch_size == 0:
					self.deltaU /= batch_size
					self.deltaV /= batch_size
					self.deltaW /= batch_size
					self.apply_deltas(learning_rate)
			
			if len(X) % batch_size > 0:
				mod = len(X) % batch_size
				self.deltaU /= mod
				self.deltaV /= mod
				self.deltaW /= mod
				self.apply_deltas(learning_rate)
			
			loss = sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / loss_sum
			acc = sum([self.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / len(X_dev)
			
			if log:
				stdout.write("\tepoch done in %.02f seconds" % (time.time() - t0))
				stdout.write("\tnew loss: {0}".format(loss))
				stdout.write("\tnew acc: {0}".format(acc))
				stdout.flush()
			
			if loss < best_loss:
				best_loss = loss
				best_acc = acc
				bestU, bestV, bestW = self.U.copy(), self.V.copy(), self.W.copy()
				best_epoch = epoch
			
			# make sure we change the RNN enough
			if abs(prev_loss - loss) < min_change:
				min_change_count += 1
			else:
				min_change_count = 0
			if min_change_count > 2:
				print("\n\ntraining finished after {0} epochs due to minimal change in loss".format(epoch+1))
				break
			
			prev_loss = loss
		
		t = time.time() - t_start
		
		if min_change_count <= 2:
			print("\n\ntraining finished after reaching maximum of {0} epochs".format(epochs))
		print("best observed loss was {0}, acc {1}, at epoch {2}".format(best_loss, best_acc, (best_epoch+1)))
		
		print("setting U, V, W to matrices from best epoch")
		self.U, self.V, self.W = bestU, bestV, bestW
		
		return best_loss


if __name__ == "__main__":

	#mode = sys.argv[1].lower()
	mode = "train-lm"
	#data_folder = sys.argv[2]
	data_folder = "../data"
	np.random.seed(2018)
	
	if mode == "train-lm":
		'''
		code for training language model.
		change this to different values, or use it to get you started with your own testing class
		'''
		train_size = 1000
		dev_size = 1000
		vocab_size = 2000
		
		# hdim = int(sys.argv[3])
		# lookback = int(sys.argv[4])
		# lr = float(sys.argv[5])
		
		# get the data set vocabulary
		vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
		num_to_word = dict(enumerate(vocab.index[:vocab_size]))
		word_to_num = invert_dict(num_to_word)
		
		# calculate loss vocabulary words due to vocab_size
		fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
		print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))
		
		docs = load_lm_dataset(data_folder + '/wiki-train.txt')
		S_train = docs_to_indices(docs, word_to_num, 1, 1)
		X_train, D_train = seqs_to_lmXY(S_train)

		# Load the dev set (for tuning hyperparameters)
		docs = load_lm_dataset(data_folder + '/wiki-dev.txt')
		S_dev = docs_to_indices(docs, word_to_num, 1, 1)
		X_dev, D_dev = seqs_to_lmXY(S_dev)
		
		X_train = X_train[:train_size]
		D_train = D_train[:train_size]
		X_dev = X_dev[:dev_size]
		D_dev = D_dev[:dev_size]

		# q = best unigram frequency from omitted vocab
		# this is the best expected loss out of that set
		q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])
		
		##########################
		# --- your code here --- #

		# rnn = RNN(vocab_size, hdim, vocab_size)
		# run_loss = rnn.train(X_train, D_train, X_dev, D_dev)
		# adjusted_loss = adjust_loss(run_loss, fraction_lost, q, mode='basic')
		# print("Unadjusted: %.03f" % np.exp(run_loss))
		# print("Adjusted for missing vocab: %.03f" % np.exp(adjusted_loss))

		lrs = [0.5,0.1,0.05]
		hdims = [25,50]
		loopbacks = [0,2,5]
		epoch_nums = 10
		for lr in lrs:
			for lookback in loopbacks:
				for hdim in hdims:
					print('#'*10)
					print('Learning Rate: ',lr,' Number of Loop Back Steps: ',lookback,' Number of Hidden Units: ',hdim)
					train_RNN = RNN(vocab_size, hdim, vocab_size)
					run_loss = train_RNN.train(X=X_train, D=D_train, X_dev=X_dev, D_dev=D_dev, epochs=epoch_nums, learning_rate=lr, back_steps=lookback)
					print("Unadjusted: %.03f" % np.exp(run_loss))
					adjusted_loss = adjust_loss(run_loss, fraction_lost, q, mode='basic')
					print("Adjusted for missing vocab: %.03f" % np.exp(adjusted_loss))
					print('#'*10)
		##########################





	if mode == "train-np":
		'''
		starter code for parameter estimation.
		change this to different values, or use it to get you started with your own testing class
		'''
		train_size = 1000
		dev_size = 1000
		vocab_size = 2000
		
		hdim = int(sys.argv[3])
		lookback = int(sys.argv[4])
		lr = float(sys.argv[5])
		
		# get the data set vocabulary
		vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
		num_to_word = dict(enumerate(vocab.index[:vocab_size]))
		word_to_num = invert_dict(num_to_word)
		
		# calculate loss vocabulary words due to vocab_size
		fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
		print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))
		
		# load training data
		sents = load_np_dataset(data_folder + '/wiki-train.txt')
		S_train = docs_to_indices(sents, word_to_num, 0, 0)
		X_train, D_train = seqs_to_npXY(S_train)
		
		X_train = X_train[:train_size]
		Y_train = D_train[:train_size]

		# load development data
		sents = load_np_dataset(data_folder + '/wiki-dev.txt')
		S_dev = docs_to_indices(sents, word_to_num, 0, 0)
		X_dev, D_dev = seqs_to_npXY(S_dev)
		
		X_dev = X_dev[:dev_size]
		D_dev = D_dev[:dev_size]


		##########################
		# --- your code here --- #
		##########################

		acc = 0.
		
		print("Accuracy: %.03f" % acc)

	
	if mode == "predict-lm":
		
		data_folder = sys.argv[2]
		rnn_folder = sys.argv[3]

		# get saved RNN matrices and setup RNN
		U,V,W = np.load(rnn_folder + "/rnn.U.npy"), np.load(rnn_folder + "/rnn.V.npy"), np.load(rnn_folder + "/rnn.W.npy")
		vocab_size = len(V[0])
		hdim = len(U[0])

		dev_size = 1000

		r = RNN(vocab_size, hdim, vocab_size)
		r.U = U
		r.V = V
		r.W = W

		# get vocabulary
		vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
		num_to_word = dict(enumerate(vocab.index[:vocab_size]))
		word_to_num = invert_dict(num_to_word)

		# Load the dev set (for tuning hyperparameters)
		docs = load_lm_np_dataset(data_folder + '/wiki-dev.txt')
		S_np_dev = docs_to_indices(docs, word_to_num, 1, 0)
		X_np_dev, D_np_dev = seqs_to_lmnpXY(S_np_dev)

		X_np_dev = X_np_dev[:dev_size]
		D_np_dev = D_np_dev[:dev_size]

		np_acc = r.compute_acc_lmnp(X_np_dev, D_np_dev)

		print('Number prediction accuracy on dev set:', np_acc)

		# load test data
		sents = load_lm_np_dataset(data_folder + '/wiki-test.txt')
		S_np_test = docs_to_indices(sents, word_to_num, 1, 0)
		X_np_test, D_np_test = seqs_to_lmnpXY(S_np_test)

		np_acc_test = r.compute_acc_lmnp(X_np_test, D_np_test)

		print('Number prediction accuracy on test set:', np_acc_test)
