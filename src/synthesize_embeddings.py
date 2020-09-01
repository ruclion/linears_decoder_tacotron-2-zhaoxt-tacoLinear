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
os.environ["CUDA_VISIBLE_DEVICES"]="4"
def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	run_name = args.name or args.tacotron_name or args.model
	taco_checkpoint = args.output_model_path
	return taco_checkpoint, modified_hp
def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def generate_fast(model, text):
    model.synthesize(text, None, None, None, None)


def run_eval(args, checkpoint_path, output_dir, hparams,synth):



    if args.model == 'Tacotron-2':
        assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

    # Create output path if it doesn't exist



    if args.reference_audio_path is not None:
        print('reference_audio:', args.reference_audio_path)
        ref_wavs = os.listdir(args.reference_audio_path)

    else:
        if hparams.use_style_encoder == True:
            print("*******************************")
            print("TODO: add style weights when there is no reference audio. Now we use random weights, " +
                  "which may generate unintelligible audio sometimes.")
            print("*******************************")
        else:
            # raise ValueError("You must set the reference audio if you don't want to use GSTs.")
            print("233")

    # Set inputs batch wise
    counter=0
    fault_ppgs=np.zeros((1,2,345),dtype=np.float32)
    for ref_wav in ref_wavs:
        speaker = ref_wav.split('_')[0]
        sentence = ref_wav.split('_')[1]
        save_path = output_dir+'/'+speaker+'/'+sentence
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        counter=counter+1
        ref_wav_name = os.path.join(args.reference_audio_path,ref_wav)
        save_name = os.path.join(save_path+'/'+ref_wav.split('.')[0]+'.npy')
        ref_wav = load_wav(ref_wav_name, hparams.sample_rate)
        reference_mel = melspectrogram(ref_wav, hparams).astype(np.float32).T
        style_embedding = synth.synthesize_embedding(fault_ppgs,[reference_mel])[0]
        np.save(save_name,style_embedding)
        print(str(counter)+'/'+str(len(ref_wavs)))







def embedding_synthesize(args, hparams, checkpoint, ppgs=None, speakers=None):
    output_dir = args.output_dir

    try:
        checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
        # checkpoint_path = '/home/zhaoxt20/vae_tac_myself/exp_multi_2020.4.1_2DPPgs+ref_same_speaker_dif_sentence/pretrained_model/tacotron_model.ckpt-45000'
        log('loaded model at {}'.format(checkpoint_path))
    except:
        raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

    if hparams.tacotron_synthesis_batch_size < hparams.tacotron_num_gpus:
        raise ValueError(
            'Defined synthesis batch size {} is smaller than minimum required {} (num_gpus)! Please verify your synthesis batch size choice.'.format(
                hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

    if hparams.tacotron_synthesis_batch_size % hparams.tacotron_num_gpus != 0:
        raise ValueError(
            'Defined synthesis batch size {} is not a multiple of {} (num_gpus)! Please verify your synthesis batch size choice!'.format(
                hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

    synth = Synthesizer()
    synth.load(checkpoint_path, hparams, reference_mels=True)

    return run_eval(args, checkpoint_path, output_dir, hparams,synth)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', default='',
    	help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--output-model-path', dest='output_model_path', required=True, type=str,
    	default=os.path.dirname(os.path.realpath(__file__)), help='Philly model output path.')

    parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
    parser.add_argument('--tacotron_name', help='Name of logging directory of Tacotron. If trained separately')
    parser.add_argument('--model', default='Tacotron')
    parser.add_argument('--reference_audio_path')
    parser.add_argument('--speaker_id', default=0)
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')

    #parser.add_argument('--speaker_id', default=4, help='Defines the speakers ids to use when running standalone Wavenet on a folder of mels. this variable must be a comma-separated list of ids')

    args = parser.parse_args()
    accepted_models=['Tacotron']
    if args.model not in accepted_models:
        raise ValueError('please enter a valid model to synthesize with: {}'.format(accepted_models))

    taco_checkpoint, hparams = prepare_run(args)

    speaker_id = 0

    #speaker_ids=[]

    if args.model == 'Tacotron':
        _ = embedding_synthesize(args, hparams, taco_checkpoint)
    else:
        raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))

if __name__ =='__main__':
    main()
