import os
import threading
import time
import traceback
import random
import numpy as np
import tensorflow as tf
from infolog import log
from sklearn.model_selection import train_test_split
#from tacotron.utils.text import text_to_sequence
import zipfile
import io

_batches_per_group = 16

class Feeder:
	"""
		Feeds batches of data into queue on a background thread.
	"""

	def __init__(self, coordinator, metadata_filename, hparams):
		super(Feeder, self).__init__()
		self._coord = coordinator
		self._hparams = hparams
		self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		self._train_offset = 0
		self._test_offset = 0
		self.hparams_1 = hparams
		#生成说话人-wav对，以找到同说话人不同内容的音频
		#self.speaker_wavs = self._get_LibriTTS_wavs()
		# Load metadata
		#self._mel_dir = os.path.join(os.path.dirname(metadata_filename), 'mels')
		#self._linear_dir = os.path.join(os.path.dirname(metadata_filename), 'linear')
		self._mel_zip = zipfile.ZipFile(os.path.dirname(metadata_filename)+"/mels.zip", "r")
		self._mel_all_zip = zipfile.ZipFile(os.path.dirname(metadata_filename)+"/mels_all.zip", "r")
		self._linear_zip = zipfile.ZipFile(os.path.dirname(metadata_filename)+"/linear.zip", "r")

		with open(metadata_filename, encoding='utf-8') as f:
			self._metadata = [line.strip().split('%') for line in f]
			frame_shift_ms = hparams.hop_size / hparams.sample_rate
			print(self._metadata[1])
			hours = sum([int(x[4]) for x in self._metadata]) * frame_shift_ms / (3600)
			log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(self._metadata), hours))

		#Train test split
		if hparams.tacotron_test_size is None:
			assert hparams.tacotron_test_batches is not None

		test_size = (hparams.tacotron_test_size if hparams.tacotron_test_size is not None
			else hparams.tacotron_test_batches * hparams.tacotron_batch_size)
		indices = np.arange(len(self._metadata))
		train_indices, test_indices = train_test_split(indices,
			test_size=test_size, random_state=hparams.tacotron_data_random_state)

		#Make sure test_indices is a multiple of batch_size else round up
		len_test_indices = self._round_down(len(test_indices), hparams.tacotron_batch_size)
		extra_test = test_indices[len_test_indices:]
		test_indices = test_indices[:len_test_indices]
		train_indices = np.concatenate([train_indices, extra_test])

		self._train_meta = list(np.array(self._metadata)[train_indices])
		self._test_meta = list(np.array(self._metadata)[test_indices])

		self.test_steps = len(self._test_meta) // hparams.tacotron_batch_size

		if hparams.tacotron_test_size is None:
			assert hparams.tacotron_test_batches == self.test_steps

		#pad input sequences with the <pad_token> 0 ( _ )
		self._pad = 0
		#explicitely setting the padding to a value that doesn't originally exist in the spectogram
		#to avoid any possible conflicts, without affecting the output range of the model too much
		if hparams.symmetric_mels:
			self._target_pad = -hparams.max_abs_value
		else:
			self._target_pad = 0.
		#Mark finished sequences with 1s
		self._token_pad = 1.

		with tf.device('/cpu:0'):
			# Create placeholders for inputs and targets. Don't specify batch size because we want
			# to be able to feed different batch sizes at eval time.
			self._placeholders = [
			tf.placeholder(tf.float32, shape=(None, None,hparams.PPGs_length), name='inputs'),
			tf.placeholder(tf.int32, shape=(None,), name='speaker'),
			tf.placeholder(tf.int32, shape=(None, ), name='input_lengths'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='mel_targets'),
			tf.placeholder(tf.float32, shape=(None, None), name='token_targets'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_freq), name='linear_targets'),
			tf.placeholder(tf.int32, shape=(None, ), name='targets_lengths'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='reference_mels'),
			tf.placeholder(tf.int32, shape=(None,), name='reference_lengths'),
			tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='split_infos'),
			tf.placeholder(tf.float32, shape=(None, None,2), name='Lf0')
			]

			# Create queue for buffering data
			queue = tf.FIFOQueue(8, [tf.float32, tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.int32, tf.int32,tf.float32], name='input_queue')
			self._enqueue_op = queue.enqueue(self._placeholders)
			self.inputs, self.speaker, self.input_lengths, self.mel_targets, self.token_targets, self.linear_targets, self.targets_lengths, self.reference_mels, self.reference_lengths, \
				 self.split_infos,self.Lf0 = queue.dequeue()

			self.inputs.set_shape(self._placeholders[0].shape)
			self.speaker.set_shape(self._placeholders[1].shape)
			self.input_lengths.set_shape(self._placeholders[2].shape)
			self.mel_targets.set_shape(self._placeholders[3].shape)
			self.token_targets.set_shape(self._placeholders[4].shape)
			self.linear_targets.set_shape(self._placeholders[5].shape)
			self.targets_lengths.set_shape(self._placeholders[6].shape)
			self.reference_mels.set_shape(self._placeholders[7].shape)
			self.reference_lengths.set_shape(self._placeholders[8].shape)
			self.split_infos.set_shape(self._placeholders[9].shape)
			self.Lf0.set_shape(self._placeholders[10].shape)

			# Create eval queue for buffering eval data
			eval_queue = tf.FIFOQueue(1, [tf.float32, tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.int32, tf.int32,tf.float32], name='eval_queue')
			self._eval_enqueue_op = eval_queue.enqueue(self._placeholders)
			self.eval_inputs, self.eval_speaker, self.eval_input_lengths, self.eval_mel_targets, self.eval_token_targets, \
				self.eval_linear_targets, self.eval_targets_lengths, self.eval_reference_mels, self.eval_reference_lengths, self.eval_split_infos,self.eval_Lf0 = eval_queue.dequeue()

			self.eval_inputs.set_shape(self._placeholders[0].shape)
			self.eval_speaker.set_shape(self._placeholders[1].shape)
			self.eval_input_lengths.set_shape(self._placeholders[2].shape)
			self.eval_mel_targets.set_shape(self._placeholders[3].shape)
			self.eval_token_targets.set_shape(self._placeholders[4].shape)
			self.eval_linear_targets.set_shape(self._placeholders[5].shape)
			self.eval_targets_lengths.set_shape(self._placeholders[6].shape)
			self.eval_reference_mels.set_shape(self._placeholders[7].shape)
			self.eval_reference_lengths.set_shape(self._placeholders[8].shape)
			self.eval_split_infos.set_shape(self._placeholders[9].shape)
			self.eval_Lf0.set_shape(self._placeholders[10].shape)

	def start_threads(self, session):
		self._session = session
		thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

		thread = threading.Thread(name='background', target=self._enqueue_next_test_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

	def _get_test_groups(self):
		meta = self._test_meta[self._test_offset]
		self._test_offset += 1

		ppgs =(np.load(meta[6]))
		
		#ppgs = np.argmax(ppgs,axis=1)
		input_data = np.asarray(ppgs)
		#input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
		speaker_id = int(meta[7])
		Lf0 = np.load(meta[8])
		
		#论文方法
		# std = 0.25 * np.std(Lf0,axis=0)+0.00001 #防止除0
		std = 4 * np.std(Lf0,axis=0)+0.00001 #防止除0
		std[1]=1
		avg = np.mean(Lf0,axis=0)
		avg[1]=0
		Lf0 = (Lf0 - avg) / std 
		
		#mel_target = np.load(os.path.join(self._mel_dir, meta[1]))
		mel_target = self._read_npy_from_zip(self._mel_zip, 'mels/' + meta[1])

		if self.hparams_1.dataset =='LibriTTS':#这个数据集是特别生成的，因此要专门处理它的同说话人不同语句的例子

			ref_name = meta[3]
			#ref_name = ref_name[:-4]+'.npy'#不要.wav
			ref_target = self._read_npy_from_zip(self._mel_all_zip, 'mels/' + 'mel-'+ref_name+'.npy')

			
		else:
			ref_target=mel_target
		#Create parallel sequences containing zeros to represent a non finished sequence
		token_target = np.asarray([0.] * (len(mel_target) - 1))
		#linear_target = np.load(os.path.join(self._linear_dir, meta[2]))
		linear_target = self._read_npy_from_zip(self._linear_zip,'linear/' + meta[2])
		# if ppgs.shape[0]>2000:
		# 	ppgs = ppgs[:int(ppgs.shape[0]*0.5),:]#In case OOM？
		# 	mel_target = mel_target[:int(mel_target.shape[0]*0.5),:]
		# 	linear_target = linear_target[:int(linear_target.shape[0]*0.5),:]
		# 	Lf0 = Lf0[:int(len(Lf0)*0.5)]
		return (input_data, speaker_id, mel_target, token_target, linear_target, len(mel_target), ref_target, len(ref_target),Lf0)

	def make_test_batches(self):
		start = time.time()

		# Read a group of examples
		n = self._hparams.tacotron_batch_size
		r = self._hparams.outputs_per_step

		#Test on entire test set
		examples = [self._get_test_groups() for i in range(len(self._test_meta))]

		# Bucket examples based on similar output sequence length for efficiency
		examples.sort(key=lambda x: x[-4])
		batches = [examples[i: i+n] for i in range(0, len(examples), n)]
		np.random.shuffle(batches)
		#batches = batches[20:]#会不会test过多
		log('\nGenerated {} test batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
		return batches, r

	def _enqueue_next_train_group(self):
		while not self._coord.should_stop():
			start = time.time()

			# Read a group of examples
			n = self._hparams.tacotron_batch_size
			r = self._hparams.outputs_per_step
			examples = [self._get_next_example() for i in range(n * _batches_per_group)]

			# Bucket examples based on similar output sequence length for efficiency
			examples.sort(key=lambda x: x[-4])
			batches = [examples[i: i+n] for i in range(0, len(examples), n)]
			np.random.shuffle(batches)

			log('\nGenerated {} train batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
			for batch in batches:
				feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r)))
				self._session.run(self._enqueue_op, feed_dict=feed_dict)

	def _enqueue_next_test_group(self):
		#Create test batches once and evaluate on them for all test steps
		test_batches, r = self.make_test_batches()
		while not self._coord.should_stop():
			for batch in test_batches:
				feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r)))
				self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

	def _get_next_example(self):
		"""Gets a single example (input, mel_target, token_target, linear_target, mel_length) from_ disk
		"""
		if self._train_offset >= len(self._train_meta):
			self._train_offset = 0
			np.random.shuffle(self._train_meta)

		meta = self._train_meta[self._train_offset]
		self._train_offset += 1

		ppgs =(np.load(meta[6]))
		
		input_data = np.asarray(ppgs)
		#input_data = np.asarray(text_to_sequence(text, self._cleaner_names), dtype=np.int32)
		speaker_id = int(meta[7])
		Lf0 = np.load(meta[8])
		
		#论文归一化
		#std = 0.25 * np.std(Lf0,axis=0)+0.00001
		std = 4 *  np.std(Lf0,axis=0)+0.00001 #防止除0
		std[1]=1
		avg = np.mean(Lf0,axis=0)
		avg[1]=0
		Lf0 = (Lf0 - avg) / std 

		#Lf0 = Lf0[:,np.newaxis]
		#mel_target = np.load(os.path.join(self._mel_dir, meta[1]))
		mel_target = self._read_npy_from_zip(self._mel_zip, 'mels/' + meta[1])

		if self.hparams_1.dataset =='LibriTTS':#这个数据集是特别生成的，因此要专门处理它的同说话人不同语句的例子

			ref_name = meta[3]
			#ref_name = ref_name[:-4]+'.npy'#不要.wav
			ref_target = self._read_npy_from_zip(self._mel_all_zip, 'mels/' + 'mel-'+ref_name+'.npy')

		else:
			ref_target = mel_target
		#Create parallel sequences containing zeros to represent a non finished sequence
		token_target = np.asarray([0.] * (len(mel_target) - 1))
		#linear_target = np.load(os.path.join(self._linear_dir, meta[2]))
		linear_target = self._read_npy_from_zip(self._linear_zip, 'linear/' + meta[2])
		
		return (input_data, speaker_id, mel_target, token_target, linear_target, len(mel_target), ref_target, len(ref_target),Lf0)

	def _prepare_batch(self, batches, outputs_per_step):
		assert 0 == len(batches) % self._hparams.tacotron_num_gpus
		size_per_device = int(len(batches) / self._hparams.tacotron_num_gpus)
		np.random.shuffle(batches)

		inputs = None
		Lf0s=None
		mel_targets = None
		token_targets = None
		linear_targets = None
		ref_targets=None
		targets_lengths = None
		ref_targets_lengths=None
		split_infos = []

		speaker = np.asarray([x[1] for x in batches], dtype=np.int32)
		targets_lengths = np.asarray([x[-4] for x in batches], dtype=np.int32) #Used to mask loss
		ref_targets_lengths=np.asarray([x[-2] for x in batches], dtype=np.int32)
		input_lengths = np.asarray([len(x[0]) for x in batches], dtype=np.int32)

		for i in range(self._hparams.tacotron_num_gpus):
			batch = batches[size_per_device*i:size_per_device*(i+1)]
			input_cur_device, input_max_len = self._prepare_inputs([x[0] for x in batch])
			inputs = np.concatenate((inputs, input_cur_device), axis=1) if inputs is not None else input_cur_device
			#Lf0 is same as ppg input so may have problem？
			Lf0_cur_device, Lf0_max_len = self._prepare_F0_inputs([x[8] for x in batch],input_max_len)

			Lf0s = np.concatenate((Lf0s, Lf0_cur_device), axis=1) if Lf0s is not None else Lf0_cur_device

			mel_target_cur_device, mel_target_max_len = self._prepare_targets([x[2] for x in batch], outputs_per_step)
			mel_targets = np.concatenate(( mel_targets, mel_target_cur_device), axis=1) if mel_targets is not None else mel_target_cur_device
			#这里改成ref，将其维度变成一样的
			ref_target_cur_device, ref_target_max_len = self._prepare_targets([x[6] for x in batch], outputs_per_step)
			ref_targets = np.concatenate((ref_targets, ref_target_cur_device), axis=1) if ref_targets is not None else ref_target_cur_device

			#Pad sequences with 1 to infer that the sequence is done
			token_target_cur_device, token_target_max_len = self._prepare_token_targets([x[3] for x in batch], outputs_per_step)
			token_targets = np.concatenate((token_targets, token_target_cur_device),axis=1) if token_targets is not None else token_target_cur_device
			linear_targets_cur_device, linear_target_max_len = self._prepare_targets([x[4] for x in batch], outputs_per_step)
			linear_targets = np.concatenate((linear_targets, linear_targets_cur_device), axis=1) if linear_targets is not None else linear_targets_cur_device
			split_infos.append([input_max_len, mel_target_max_len, token_target_max_len, linear_target_max_len, ref_target_max_len,Lf0_max_len])

		split_infos = np.asarray(split_infos, dtype=np.int32)
		#这里ref本来是mel
		return (inputs, speaker, input_lengths, mel_targets, token_targets, linear_targets, targets_lengths, ref_targets, ref_targets_lengths, split_infos,Lf0s)

	def _prepare_F0_inputs(self, inputs,ppg_max):
		temp = [len(x) for x in inputs]
		max_len = max([len(x) for x in inputs])
		if max_len < ppg_max:
			max_len = max_len+1
		temp1 = np.stack([self._pad_input(x, max_len) for x in inputs])
		return temp1, max_len

	def _prepare_inputs(self, inputs):
		temp = [len(x) for x in inputs]
		max_len = max([len(x) for x in inputs])
		
		temp1 = np.stack([self._pad_input(x, max_len) for x in inputs])
		return temp1, max_len

	def _prepare_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets])
		data_len = self._round_up(max_len, alignment)
		return np.stack([self._pad_target(t, data_len) for t in targets]), data_len

	def _prepare_token_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets]) + 1
		data_len = self._round_up(max_len, alignment)
		return np.stack([self._pad_token_target(t, data_len) for t in targets]), data_len

	def _pad_input(self, x, length):
		
		temp = np.pad(x, ((0, length - x.shape[0]),(0,0)), mode='constant', constant_values=self._pad)
		return temp

	def _pad_target(self, t, length):
		return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

	def _pad_token_target(self, t, length):
		return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=self._token_pad)

	def _round_up(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x + multiple - remainder

	def _round_down(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x - remainder

	def _read_npy_from_zip(self, zfile, name):
		zip_npy = zfile.open(name, 'r')
		raw_npy = io.BytesIO(zip_npy.read())
		return np.load(raw_npy)

	# def _get_LibriTTS_wavs(self):#生成一个说话人音频的列表，以找到同说话人不同句的结果 2020.3.25
	# 	self.split_dir = '/home/zhaoxt20/vae_tac_myself/LibriTTS_training_data/train.txt'
	# 	speaker_wavs = {}
	# 	with open(self.split_dir,'r') as f:
	# 		lines = f.readlines()

	# 	for line in lines:
	# 		speaker = line.split('%')[0].split('-')[1].split('_')[0]
	# 		if speaker not in speaker_wavs:
	# 			speaker_wavs[speaker] = []
	# 			speaker_wavs[speaker].append(line.split('%')[0].split('-')[1])
	# 		else:
	# 			speaker_wavs[speaker].append(line.split('%')[0].split('-')[1])
	# 	return speaker_wavs


if __name__=='__main__':
	Feeder()