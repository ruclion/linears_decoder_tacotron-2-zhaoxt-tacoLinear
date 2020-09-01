import argparse
import os
from time import sleep

import infolog
import tensorflow as tf
from hparams import hparams
from infolog import log
from tacotron.synthesize import tacotron_synthesize
from tacotron.train import tacotron_train

log = infolog.log


def save_seq(file, sequence, input_path):
	'''Save Tacotron-2 training state to disk. (To skip for future runs)
	'''
	sequence = [str(int(s)) for s in sequence] + [input_path]
	with open(file, 'w') as f:
		f.write('|'.join(sequence))

def read_seq(file):
	'''Load Tacotron-2 training state from disk. (To skip if not first run)
	'''
	if os.path.isfile(file):
		with open(file, 'r') as f:
			sequence = f.read().split('|')

		return [bool(int(s)) for s in sequence[:-1]], sequence[-1]
	else:
		return [0, 0, 0], ''

def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
	run_name = args.name or args.model
	log_dir = args.log_dir
	os.makedirs(log_dir, exist_ok=True)
	infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name, args.slack_url)
	return log_dir, modified_hp


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--output-model-path', dest='output_model_path', required=True, type=str,
		default=os.path.dirname(os.path.realpath(__file__)), help='Philly model output path.')
	parser.add_argument('--log-dir', dest='log_dir', required=True, type=str,
		default=os.path.dirname(os.path.realpath(__file__)), help='Philly log dir.')
	parser.add_argument('--tacotron_input', default='clean100/training_data/train.txt')
	parser.add_argument('--name', help='Name of logging directory.')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
	parser.add_argument('--summary_interval', type=int, default=100,
		help='Steps between running summary ops')
	parser.add_argument('--embedding_interval', type=int, default=10000,
		help='Steps between updating embeddings projection visualization')
	parser.add_argument('--checkpoint_interval', type=int, default=5000,
		help='Steps between writing checkpoints')
	parser.add_argument('--eval_interval', type=int, default=1000,
		help='Steps between eval on test data')
	parser.add_argument('--tacotron_train_steps', type=int, default=150000, help='total number of tacotron training steps')
	parser.add_argument('--wavenet_train_steps', type=int, default=750000, help='total number of wavenet training steps')
	parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
	parser.add_argument('--slack_url', default=None, help='slack webhook notification destination link')
	args = parser.parse_args()

	accepted_models = ['Tacotron']

	if args.model not in accepted_models:
		raise ValueError('please enter a valid model to train: {}'.format(accepted_models))

	log_dir, hparams = prepare_run(args)

	if args.model == 'Tacotron':
		hparams.tacotron_batch_size = 16
		tacotron_train(args, log_dir, hparams)
	else:
		raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))


if __name__ == '__main__':
	main()
