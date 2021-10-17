from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset_custom import mel_spectrogram, MAX_WAV_VALUE, load_wav
from model_mel80 import Generator
import numpy as np
import time
h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    #filelist = os.listdir(a.input_wavs_dir)LJ004-0134-feats.npy
    filelist = sorted(glob.glob(os.path.join(a.input_dir, '*.npy')))


    #filelist = glob.glob(os.path.join(a.input_wavs_dir, '*.npy'))
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        times= []
        totlen =0
        tottime= 0
        for i, filname in enumerate(filelist):
            x = torch.from_numpy(np.load(filname)).T.unsqueeze(0).cuda()
#             x = torch.from_numpy(np.load(filname)).unsqueeze(0).to(device)  # (80, T)
       
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            # audio = audio / np.abs(audio).max()

            output_file = os.path.join(a.output_dir, filname.split('/')[-1] + '.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/hd0/hs_oh/dataset/kss/mel/')
    parser.add_argument('--output_dir', default='/hd0/hs_oh/dataset/kss/temp')
    parser.add_argument('--checkpoint_file', default='./g_02400000')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

