#!/usr/bin/bash
# usage bash wav2lf0.sh /path/to/wav /path/to/save/lf0
wav_dir=$1
lf0_dir=$2

echo "compute lf0 from wavs in $wav_dir and save to $lf0_dir"

F0_low=30
F0_high=500
frame_shift=80
sample_rate_khz=16

for f in $(find $wav_dir -type f -name "*.wav");do
	sox $f -t raw temp.raw;
	fn=$(basename "$f" .wav);
	x2x +sf temp.raw | pitch -H $F0_high -L $F0_low -p $frame_shift -s $sample_rate_khz -o 2 > $lf0_dir/$fn.lf0;
	rm temp.raw;
done