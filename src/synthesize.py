import argparse
import os
from warnings import warn
from time import sleep

import tensorflow as tf
import numpy as np
from hparams import hparams
from infolog import log
from tacotron.synthesize import tacotron_synthesize
import speakers_clean100 as speakers


os.environ["CUDA_VISIBLE_DEVICES"]=" "
def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	run_name = args.name or args.tacotron_name or args.model
	taco_checkpoint = args.output_model_path
	return taco_checkpoint, modified_hp


def get_ppgs(args):
	ppgs=[]
	if args.ppg_path != '':
		test_root = os.listdir(args.ppg_path)
		test_root.sort()
		test_root=test_root[:100]
		for x in test_root:
			ppg_name = os.path.join(args.ppg_path,x)
			ppg = np.load(ppg_name)
			#ppg = np.argmax(ppg,axis=1)
			ppgs.append(ppg)
	else:
		ppgs = hparams.sentences

	return np.asarray(ppgs)

def get_Lf0s(args):
	Lf0s=[]
	if args.lf0_path != '':
		test_root = os.listdir(args.lf0_path)
		test_root.sort()
		test_root=test_root[:100]
		for x in test_root:
			lf0_name = os.path.join(args.lf0_path,x)
			Lf0 = np.load(lf0_name)
			#论文归一化
			std = 4 * np.std(Lf0,axis=0)+0.00001
			std[1]=1
			avg = np.mean(Lf0,axis=0)
			avg[1]=0
			Lf0 = (Lf0 - avg) / std
			Lf0s.append(Lf0)
	else:
		exit()

	return np.asarray(Lf0s)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--output-model-path', dest='output_model_path', required=True, type=str,
		default=os.path.dirname(os.path.realpath(__file__)), help='Philly model output path.')

	parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
	parser.add_argument('--tacotron_name', help='Name of logging directory of Tacotron. If trained separately')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--reference_audio')
	parser.add_argument('--speaker_id', default=0)
	parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--ppg_path', default='', help='Dir which contains test ppgs.')
	parser.add_argument('--lf0_path', default='', help='Dir which contains lf0.')
	#parser.add_argument('--speaker_id', default=4, help='Defines the speakers ids to use when running standalone Wavenet on a folder of mels. this variable must be a comma-separated list of ids')
	args = parser.parse_args()

	accepted_models = ['Tacotron']

	if args.model not in accepted_models:
		raise ValueError('please enter a valid model to synthesize with: {}'.format(accepted_models))

	taco_checkpoint, hparams = prepare_run(args)


	ppgs = get_ppgs(args)
	Lf0s = get_Lf0s(args)
	if args.reference_audio:
		audioName = args.reference_audio.split('/')[-1]
	if hparams.use_multispeaker:
		if hparams.dataset == "LibriTTS":
			# for libritts
			speaker_id = speakers.speaker2Id[int(audioName[0: audioName.find('_')])]
		if hparams.dataset.startswith("multi"):
			speaker_id = args.speaker_id
	else:
		speaker_id = 0
	speaker_ids = [speaker_id]*len(ppgs)
	#speaker_ids=[]
	if args.model == 'Tacotron':
		print(args.reference_audio)
		#exit()
		_ = tacotron_synthesize(args, hparams, taco_checkpoint, ppgs, speaker_ids,Lf0s)
	else:
		raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))


if __name__ == '__main__':
	main()
