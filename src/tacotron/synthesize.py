import argparse
import os
import re
import time
from time import sleep

import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
from tacotron.synthesizer import Synthesizer
from tqdm import tqdm
from datasets.audio import load_wav, melspectrogram
import numpy as np

def get_output_base_path(checkpoint_path):
	base_dir = os.path.dirname(checkpoint_path)
	m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
	name = 'eval-%d' % int(m.group(1)) if m else 'eval'
	return os.path.join(base_dir, name)
def generate_fast(model, text):
	model.synthesize(text, None, None, None, None)



def run_eval(args, checkpoint_path, output_dir, hparams, ppgs, speakers,Lf0s):
	eval_dir = os.path.join(output_dir, 'eval')
	log_dir = os.path.join(output_dir, 'logs-eval')

	if args.model == 'Tacotron-2':
		assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

	#Create output path if it doesn't exist
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams, reference_mels=args.reference_audio)

	if args.reference_audio is not None:
		print('reference_audio:', args.reference_audio)
		ref_wav = load_wav(args.reference_audio.strip(), hparams.sample_rate)
		reference_mel = melspectrogram(ref_wav, hparams).astype(np.float32).T
	else:
		if hparams.use_style_encoder == True:
			print("*******************************")
			print("TODO: add style weights when there is no reference audio. Now we use random weights, " + 
			"which may generate unintelligible audio sometimes.")
			print("*******************************")
		else:
			#raise ValueError("You must set the reference audio if you don't want to use GSTs.")
			print("233")

	#Set inputs batch wise
	ppgs = [ppgs[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(ppgs), hparams.tacotron_synthesis_batch_size)]
	Lf0s = [Lf0s[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(Lf0s), hparams.tacotron_synthesis_batch_size)]
	if args.reference_audio is not None:
		reference_mels = [reference_mel]*len(ppgs)


	log('Starting Synthesis')
	with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
		
		for i, texts in enumerate(tqdm(ppgs)):
			start = time.time()
			basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
			if args.reference_audio is not None:
				mel_filenames = synth.synthesize(texts, [speakers[i]], basenames, eval_dir, log_dir, None, [reference_mels[i]],Lf0s[i])
			else:
				mel_filenames = synth.synthesize(texts, [speakers[i]], basenames, eval_dir, log_dir, None, None,Lf0s[i])


			for elems in zip(texts, mel_filenames, [speakers[i]]):
				file.write('|'.join([str(x) for x in elems]) + '\n')
	log('synthesized mel spectrograms at {}'.format(eval_dir))
	return eval_dir


def tacotron_synthesize(args, hparams, checkpoint, ppgs=None, speakers=None,Lf0s=None):
	output_dir = args.output_dir

	try:
		checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
		#checkpoint_path = '/home/zhaoxt20/vae_tac_myself/exp_multi_2020.4.1_2DPPgs+ref_same_speaker_dif_sentence/pretrained_model/tacotron_model.ckpt-45000'
		log('loaded model at {}'.format(checkpoint_path))
	except:
		raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

	if hparams.tacotron_synthesis_batch_size < hparams.tacotron_num_gpus:
		raise ValueError('Defined synthesis batch size {} is smaller than minimum required {} (num_gpus)! Please verify your synthesis batch size choice.'.format(
			hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

	if hparams.tacotron_synthesis_batch_size % hparams.tacotron_num_gpus != 0:
		raise ValueError('Defined synthesis batch size {} is not a multiple of {} (num_gpus)! Please verify your synthesis batch size choice!'.format(
			hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

	return run_eval(args, checkpoint_path, output_dir, hparams, ppgs, speakers,Lf0s)
	
