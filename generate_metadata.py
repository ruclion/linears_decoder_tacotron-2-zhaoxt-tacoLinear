import os
import numpy as np
PPG_Path='/home/zhaoxt20/vae_tac_myself/LibriTTS_training_data/ppg_extracted'
WAV_Path='/home/zhaoxt20/vae_tac_myself_F0_consist/LibriTTS_16khz/wavs'
F0_Path='/home/zhaoxt20/vae_tac_myself_F0_consist/LibriTTS_16khz/Lf0'
Meta_Path = '/home/zhaoxt20/vae_tac_myself_F0_consist/LibriTTS_16khz/metadata.csv'

def main():
    wavlist = os.listdir(WAV_Path)
    with open(Meta_Path,'w') as f:
        for wav in wavlist:
            name = wav.split('.')[0]
            ppgpath =os.path.join(PPG_Path,name+'.npy')
            lf0path = os.path.join(F0_Path,name+'.npy')
            f.write(name+'%'+ppgpath+'%'+lf0path+'\n')

if __name__ == '__main__':
        main()