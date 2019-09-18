"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import os
import sys
import math
import wavio
import time 
import torch
import random
import threading
import logging
import scipy.io as sio
from nb_SparseImageWarp import sparse_image_warp

import scipy.io.wavfile
import librosa
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

PAD = 0
N_FFT = 1024
#SAMPLE_RATE = 16000

target_dict = dict()

def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target

def get_spectrogram_feature(filepath):
    sample_rate, data = sio.wavfile.read(filepath)
    (rate, width, sig) = wavio.readwav(filepath)

    sig = sig.ravel()

    #Get Sample rate from wave. 
    SAMPLE_RATE = sample_rate

    #frame length == window length 
    #win_length = N_FFT/SAMPLE_RATE 1024/16000 = 0.64 

    #hop length = window_length - frame stride 

    stft = torch.stft(torch.FloatTensor(sig),
                        N_FFT,
                        hop_length=int(0.01*SAMPLE_RATE),
                        win_length=int(0.060*SAMPLE_RATE),
                        window=torch.hamming_window(int(0.060*SAMPLE_RATE)),
                        center=False,
                        normalized=False,
                        onesided=True)

    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5);
    amag = stft.numpy();
    feat = torch.FloatTensor(amag)
    feat = torch.FloatTensor(feat).transpose(0, 1)

    return feat


def get_spectrogram_feature_2(filepath):
    sample_rate, data = sio.wavfile.read(filepath)
    (rate, width, sig) = wavio.readwav(filepath)

    sig = sig.ravel()

    #Get Sample rate from wave. 
    SAMPLE_RATE = sample_rate

    sig_f = sig.astype(float)

    mftt = librosa.feature.melspectrogram(y=sig_f, sr= SAMPLE_RATE, n_fft=N_FFT, hop_length=int(0.01*SAMPLE_RATE),win_length=int(0.060*SAMPLE_RATE))
    print(mftt)
    print(type(mftt))
    return mftt

def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    device = spec.device
    
    y = num_rows//2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len
    
    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device), 
                         torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)


def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]
    
    for i in range(0, num_masks):        
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = random.randrange(f_zero, f_zero + f) 
        if (replace_with_zero): cloned[0][f_zero:mask_end] = 0
        else: cloned[0][f_zero:mask_end] = cloned.mean()
    
    return cloned

    
def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]
    
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero): cloned[0][:,t_zero:mask_end] = 0
        else: cloned[0][:,t_zero:mask_end] = cloned.mean()
    return cloned


def get_script(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result

class BaseDataset(Dataset):
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = get_spectrogram_feature(self.wav_paths[idx])
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)
        return feat, script

def _collate_fn(batch):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets, seq_lengths, target_lengths

class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for i in range(self.batch_size): 
                if self.index >= self.dataset_count:
                    break

                items.append(self.dataset.getitem(self.index))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataset_list[i], self.queue, self.batch_size, i))

    def start(self):
        for i in range(self.worker_size):
            self.loader[i].start()

    def join(self):
        for i in range(self.worker_size):
            self.loader[i].join()

