#!/usr/bin/bash
# usage bash wav2lf0.sh /path/to/wav /path/to/save/lf0
wav_dir=$1
sav_dir=$2

echo "trans 16khz from wavs in $wav_dir and save to $sav_dir"


for f in $(find $wav_dir -type f -name "*.wav");do
	fn=$(basename "$f" .wav);
	sox $f -r 16000 $sav_dir/$fn.wav;

done