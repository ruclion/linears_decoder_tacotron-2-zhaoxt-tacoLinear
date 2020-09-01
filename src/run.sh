#!/usr/bin/env bash
dataset="LibriTTS"
language="English1"
exp="exp_multi"

pre_hparams="dataset=${dataset},use_multispeaker=False"
train_hparams="dataset=${dataset},predict_linear=True,tacotron_num_gpus=2,tacotron_batch_size=32,outputs_per_step=3,use_style_encoder=True,use_multispeaker=False,speaker_num=3,speaker_dim=32"
test_hparams="dataset=${dataset},predict_linear=True,tacotron_num_gpus=1,tacotron_batch_size=32,outputs_per_step=3,use_style_encoder=True,use_multispeaker=False,speaker_num=3,speaker_dim=32"


src_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
code_dir=${src_dir}/../
exp_dir=${code_dir}${exp}
cd ${src_dir}



start_stage=0
stop_stage=0
. ${src_dir}/parse_option.sh || exit 1;


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
	echo "stage 0: preprocess data"

	python preprocess.py --base_dir ${code_dir} --hparams ${pre_hparams}

	cd "${code_dir}${dataset}_16khz"
	echo "zip mels"
	zip mels.zip mels/*
	echo "zip linear"
	zip linear.zip linear/*
	rm -r mels
	rm -r linear
	rm -r audio
	cd ${src_dir}
fi


if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
	echo "step 1: training the tacotron model"

	if [ ! -d ${exp_dir} ]; then
		mkdir ${exp_dir}
	fi
	cp run.sh ${code_dir}${exp}/run.sh

	if [ ${language} = "English" ]; then
		echo "select symbols_eng.py according to the dataset"
		cp tacotron/utils/symbols_eng.py tacotron/utils/symbols.py
	else
		echo "select symbols_ch.py according to the dataset"
		cp tacotron/utils/symbols_ch.py tacotron/utils/symbols.py
	fi

	CUDA_VISIBLE_DEVICES="4,5"  python train.py  --tacotron_input ${code_dir}${dataset}_16khz/train.txt \
	--output-model-path ${exp_dir}/pretrained_model --log-dir ${exp_dir}/logs-Tacotron \
	--hparams ${train_hparams} --tacotron_train_steps 100000

fi


if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
	echo "step 1: testing the tacotron model"

	if [ ${language} = "English" ]; then
		echo "select sentences_eng.phoneme.txt according to the dataset"
		cp sentences_eng.phoneme.txt sentences.phoneme.txt
	else
		echo "select sentences_ch.phoneme.txt according to the dataset"
		cp sentences_ch.phoneme.txt sentences.phoneme.txt
	fi
	
	ppg_path="/home/zhaoxt20/vae_tac_myself_F0_consist/test_datas/Librispeech/cmu_bdl_ppgs"
	target="/home/zhaoxt20/vae_tac_myself_F0_consist/target/wavs/1183_128659_000048_000001.wav"
	lf0_path="/home/zhaoxt20/vae_tac_myself_F0_consist/test_datas/Librispeech/cmu_bdl_16klf0"

	output_dir="${exp_dir}/tacotron_output"
	if [ -d ${output_dir} ]; then
		rm -r ${output_dir}
	fi
	mkdir ${output_dir}

	CUDA_VISIBLE_DEVICES=" "  python synthesize.py  --model Tacotron --ppg_path ${ppg_path}  \
	--lf0_path ${lf0_path} \
	--output-model-path ${exp_dir}/pretrained_model/ --reference_audio ${target}\
	--hparams ${test_hparams} --output_dir ${output_dir} --speaker_id 0

fi


if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
	echo "step 2: testing the tacotron model for generate speaker_embedding"

	if [ ${language} = "English" ]; then
		echo "select sentences_eng.phoneme.txt according to the dataset"
		cp sentences_eng.phoneme.txt sentences.phoneme.txt
	else
		echo "select sentences_ch.phoneme.txt according to the dataset"
		cp sentences_ch.phoneme.txt sentences.phoneme.txt
	fi


	target="/home/zhaoxt20/vae_tac_myself/LibriTTS_training_data/wavs"

	output_dir="${exp_dir}/Libritts_styles"
	if [ -d ${output_dir} ]; then
		rm -r ${output_dir}
	fi
	mkdir ${output_dir}

	CUDA_VISIBLE_DEVICES='4'  python synthesize_embeddings.py  --model Tacotron  \
	--output-model-path ${exp_dir}/pretrained_model/ --reference_audio_path ${target}\
	--hparams ${test_hparams} --output_dir ${output_dir} --speaker_id 0

fi
