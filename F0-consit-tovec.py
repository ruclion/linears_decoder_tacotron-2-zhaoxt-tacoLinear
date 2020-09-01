import numpy as np
import os
#Lf0_path = "/home/zhaoxt20/vae_tac_myself_F0_consist/LibriTTS_16khz/Lf0_origin/"
Lf0_path = "/home/zhaoxt20/vae_tac_myself_F0_consist/test_datas/Librispeech/cmu_bdl_16klf0Origin/"
c = 1
d = len(os.listdir(Lf0_path))
for Lf0_file in os.listdir(Lf0_path):
    print(str(c)+'/'+str(d))
    c=c+1
    Lf0_origin = np.fromfile(Lf0_path+Lf0_file,dtype=np.float32)
    Unvoiced_index = np.where(Lf0_origin==-1e10)[0]
    Voiced_index = np.where(Lf0_origin!=-1e10)[0]

#Unvoice 1 Voice 0
    Unvoiced_flag = np.zeros(Lf0_origin.shape)

    if len(Unvoiced_index)==0:
        Unvoiced_flag = Unvoiced_flag[:,np.newaxis]
        Lf0_origin = Lf0_origin[:,np.newaxis]
        Lf0_origin = np.concatenate((Lf0_origin,Unvoiced_flag),axis=-1)
        np.save("/home/zhaoxt20/vae_tac_myself_F0_consist/test_datas/Librispeech/cmu_bdl_16klf0/"+Lf0_file.split('.')[0]+'.npy',Lf0_origin)
        continue

    Unvoiced_flag[Unvoiced_index]=1
    Unvoiced_flag = Unvoiced_flag[:,np.newaxis]
    
    if len(Voiced_index)==0:
        Lf0_origin = Lf0_origin[:,np.newaxis]
        Lf0_origin = np.concatenate((Lf0_origin,Unvoiced_flag),axis=-1)
        np.save("/home/zhaoxt20/vae_tac_myself_F0_consist/test_datas/Librispeech/cmu_bdl_16klf0/"+Lf0_file.split('.')[0]+'.npy',Lf0_origin)
        continue

    
    

#using interp to change -1e10
    xp = Voiced_index
    fp = Lf0_origin[Voiced_index]
    y = np.interp(Unvoiced_index,xp,fp)
    Lf0_origin[Unvoiced_index] = y
    Lf0_origin = Lf0_origin[:,np.newaxis]

    Lf0_origin = np.concatenate((Lf0_origin,Unvoiced_flag),axis=-1)
    if Lf0_origin.shape[-1]!=2:
        print(Lf0_path)
    #不能用tofile，这样读出来无法变成二维
    np.save("/home/zhaoxt20/vae_tac_myself_F0_consist/test_datas/Librispeech/cmu_bdl_16klf0/"+Lf0_file.split('.')[0]+'.npy',Lf0_origin)

    
    


    
    
