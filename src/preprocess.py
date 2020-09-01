import argparse
import os
from multiprocessing import cpu_count

from datasets import preprocessor
from hparams import hparams
from tqdm import tqdm
#333

def preprocess(args, input_folders, out_dir, hparams):
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	linear_dir = os.path.join(out_dir, 'linear')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(linear_dir, exist_ok=True)
	speaker_wavs = get_LibriTTS_wavs()
	metadata = preprocessor.build_from_path(hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, speaker_wavs,tqdm=tqdm)
	write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:#223
		for m in metadata:
			f.write('%'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[5]) for m in metadata])
	timesteps = sum([int(m[4]) for m in metadata])

	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))
	print('Max input length (text chars): {}'.format(max(len(m[6].split()) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[5]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[4] for m in metadata)))

def norm_data(args, hparams):

	print('Selecting data folders..')
	supported_datasets = ['LJSpeech-1.0', 'LJSpeech-1.1', 'M-AILABS','LibriTTS','CMU']#236
	if False and hparams.dataset not in supported_datasets:
		raise ValueError('dataset value entered {} does not belong to supported datasets: {}'.format(
			hparams.dataset, supported_datasets))

	if hparams.dataset.startswith('LJSpeech') or hparams.dataset.startswith('LibriTTS') or hparams.dataset.startswith('CMU'):
		return [os.path.join(args.base_dir, hparams.dataset + '_16khz')]##这里好像有点冗余！！容易改不到

	if hparams.dataset.startswith('multi'):
		return [os.path.join(args.base_dir, data) for data in hparams.dataset.split('_')[1:]]


def run_preprocess(args, hparams):
	input_folders = norm_data(args, hparams)
	output_folder = args.base_dir + hparams.dataset + '_' + args.output
	os.makedirs(output_folder, exist_ok=True)

	preprocess(args, input_folders, output_folder, hparams)


def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--output', default='16khz')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()

	modified_hp = hparams.parse(args.hparams)

	run_preprocess(args, modified_hp)

def get_LibriTTS_wavs():#生成一个说话人音频的列表，以找到同说话人不同句的结果 2020.3.25
		split_dir = '/home/zhaoxt20/vae_tac_myself_F0_consist/LibriTTS_16khz/wavs'
		dirs = os.listdir(split_dir)
		speaker_wavs = {}

		for wav in dirs:
			speaker = wav.split('_')[0]
			if speaker not in speaker_wavs:
				speaker_wavs[speaker] = []
				speaker_wavs[speaker].append(wav.split('.')[0])#7511_102419_000029_000009
			else:
				speaker_wavs[speaker].append(wav.split('.')[0])
		return speaker_wavs
if __name__ == '__main__':
	main()
