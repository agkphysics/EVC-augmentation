import random
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from .extract import extract

## These are from Common Voice 10 train files
phones = ['a', 's', 't', 'ʊ', 'd', 'j', 'o', 'tʃ', 'l', 'dʒ', 'i', 'ɔ', 'r', 'e', 'f', 'z', 'b', 'k', 'ɛ', 'n', 'm', 'p', 'ts', 'ɪ', 'ss', 'kː', 'tː', 'eɪ', 'v', 'ɾ', 'pː', 'bː', 'tʃː', 'u', 'ŋ', 'ɡ', 'aː', 'w', 'ɲ', 'ʃ', 'iː', 'ʎ', 'dʒː', 'dzː', 'uː', 'ʌ', 'ɯ', 'ɐ', 'ʌʌ', 'tsː', 'aɪ', 'dz', 'dː', 'eː', 'd̪', 'ə', 'əː', 'h', 'ɛː', 'oː', 'x', 'ʁ', 'y', 'ɔː', 'ɒ', 'oɪ', 'aʊ', 'ɹ', 't̪', 'aɪə', 'iə', 'c', 'əl', 'ɡː', 'ɜː', 'ʒ', 'θ', 'ɪː', 'ɜ', 'ɛɪ', 'uɪ', 'eə', 'ɵ', 'ʊː', 'ɣ', 'ɐɐ', 't̪ː', 'vʲ', 'ɑː', 'kh', 'əʊ', 'ɛɛ', 'œ', 'ʔ', 'dˤ', 'ʕ', 's̪', 'a.', 'ħ', 'χ', 'a.ː', 'ð', 'q', 's̪ː', 'i.', 'u.', 'qː', 'i.ː', 'dˤdˤ', 'u.ː', 'lː', 'mː', 'ç', 'dʰ', 'kʰ', 'bʰ', 'ʈ', 'ɟ', 'tʃʰ', 'æ', 'kʰː', 'tʰ', 'ʈʰ', 'r.', 'ɡʰ', 'ɖ', 'ã', 'tʰː', 'ɟʰ', 'ũ', 'õ', 'dʰː', 'ɖʰ', 'ɟː', 'ĩ', 'ʈʈ', 'ɡʰː', 'ẽ', 'ɖʰɖʰ', 'ɔ̃', 'ɳ', 'bʰː', 'cʰ', 'd^', 's^', 'ɑ', 't^', 'øː', 'æː', 'ɵː', 'yː', 'ø', 'æi', 'yi', 't^ː', 'æiː', 'øi', '1', 'rr', 'ɚ', 'oʊ', 'ɑːɹ', 'ᵻ', 'ɔːɹ', 'ɔɪ', 'ʊɹ', 'oːɹ', 'aɪɚ', 'ɛɹ', 'n̩', 'ɪɹ', 'ɬ', 'ɑ̃', 'nʲ', 'ɡʲ', 'ɐ̃', 'ɨ', 'ɐ̃ʊ̃', 'eʊ', 'iʊ', 'ɛʊ', 'sʲ', 'ʂ', 'fʲ', 'ʊə', 'õː', 'ʋ', 'ũː', 'ãː', 'ʐ', 'ẽː', 'pʰ', 'ĩː', 'ʌ̃', 'cː', 'oːː', 'eːː', 'th', 'ɑ5', 'ə1', 'a5', 'oɜ', 's.', 'ə5', 'iɜ', 'ts.h', 'i.ɜ', 'ɕ', 'y5', 'i5', 'i2', 'tɕh', 'yæɜ', 'i.5', 'tɕ', 'iɛ5', 'o5', 'ts.', 'onɡ5', 'iɛɜ', 'aiɜ', 'ɑuɜ', 'uɜ', 'uei5', 'u2', 'iou2', 'o1', 'onɡ2', 'i̪5', 'i̪1', 'y2', 'əɜ', 'iou5', 'iɑ5', 'uo5', 'iɛ2', 'iouɜ', 'ɑɜ', 'uo2', 'tsh', 'ai2', 'yu5', 'ai5', 'o2', 'ɑ2', 'ph', 'u5', 'ɑu2', 'i̪2', 'ou5', 'ʲ', 'yɜ', 'ou2', 'u1', 'ə2', 'yæ5', 'uaɜ', 'a2', 'ɑu5', 'onɡɜ', 'yɛ2', 'aɜ', 'ouɜ', 'uəɜ', 'iɑ2', 'a1', 'yiɜ', 'ei5', 'uei2', 'ər2', 'eiɜ', 'ei2', 'i.2', 'yɛ5', 'ɑ1', 'iou1', 'iɑɜ', 'ua2', 'uə5', 'ua5', 'ər5', 'uoɜ', 'yɛɜ', 'i1', 'yə5', 'ərɜ', 'ueiɜ', 'uai5', 'yæ2', 'yəɜ', 'yu2', 'i̪ɜ', 'uə2', 'uaiɜ', 'uo1', 'əː1', 'uei1', 'iɛ1', 'ua1', 'i.1', 'yə2', 'iɑ1', 'y1', 'ou1', 'ei1', 'uai2', 'uə1', 'u5ʲ', 'ər1', 'onɡ1', 'uoɜʲ', 'yɛ5ʲ', 'ɑu2ʲ', 'uɜʲ', 'u2ʲ', 'SOS/EOS']

ph2id = {ph:i for i, ph in enumerate(phones)}
id2ph = phones

emotions = ['Neutral','Happy','Sad','Angry','Surprise']

emo2id = {emo: i for i, emo in enumerate(emotions)}
id2emo = emotions


class TextMelIDLoader(torch.utils.data.Dataset):
    def __init__(self, list_file, mean_std_file, phones_csv, emo_csv, emo_list, shuffle=True, pids=None):
        file_path_list = []
        with open(list_file) as f:
            file_path_list = list(map(str.strip, f))

        df = pd.read_csv(phones_csv, header=0, index_col=0, dtype=str)
        self.phones = df.squeeze().str.strip()
        df = pd.read_csv(emo_csv, header=0, index_col=0, dtype=str)
        self.emotions = df.squeeze()
        self.emo_list = list(emo_list)

        if pids:
            file_path_list = [x for x in file_path_list if self.emotions[Path(x).stem] in pids]

        random.seed(1234)
        if shuffle:
            random.shuffle(file_path_list)

        self.file_path_list = file_path_list

        self.mel_mean_std = np.float32(np.load(mean_std_file))

    def get_text_mel_id_pair(self, path):
        '''
        text_input [len_text]
        text_targets [len_mel]
        mel [mel_bin, len_mel]
        emotion_id [1]
        '''

        emotion_id = self.emotions[Path(path).stem]

        text_input = self.phones[Path(path).stem].split()
        text_input = [ph2id[x] for x in text_input if x in ph2id]

        mel, spec = extract(path, trim=True, normalise=True)

        # Normalize audio
        mel = (mel - self.mel_mean_std[0])/ self.mel_mean_std[1]

        # Format for pytorch
        text_input = torch.LongTensor(text_input)
        mel = torch.from_numpy(np.transpose(mel))
        spec = torch.from_numpy(np.transpose(spec))
        emotion_id = torch.LongTensor([self.emo_list.index(emotion_id)])

        return (text_input, mel, spec, emotion_id)

    def __getitem__(self, index):
        return self.get_text_mel_id_pair(self.file_path_list[index])

    def __len__(self):
        return len(self.file_path_list)


class TextMelIDCollate():
    def __init__(self, n_frames_per_step=2):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        '''
        batch is list of (text_input, mel, spc, emotion_id)
        '''

        text_lengths = torch.IntTensor([len(x[0]) for x in batch])
        mel_lengths = torch.IntTensor([x[1].size(1) for x in batch])
        mel_bin = batch[0][1].size(0)
        spc_bin = batch[0][2].size(0)

        max_text_len = torch.max(text_lengths).item()
        max_mel_len = torch.max(mel_lengths).item()
        if max_mel_len % self.n_frames_per_step != 0:
            max_mel_len += self.n_frames_per_step - max_mel_len % self.n_frames_per_step
            assert max_mel_len % self.n_frames_per_step == 0

        text_input_padded = torch.LongTensor(len(batch), max_text_len)
        mel_padded = torch.FloatTensor(len(batch), mel_bin, max_mel_len)
        spc_padded = torch.FloatTensor(len(batch), spc_bin, max_mel_len)

        emotion_id = torch.LongTensor(len(batch))
        stop_token_padded = torch.FloatTensor(len(batch), max_mel_len)

        text_input_padded.zero_()
        mel_padded.zero_()
        spc_padded.zero_()
        emotion_id.zero_()
        stop_token_padded.zero_()

        for i in range(len(batch)):
            text =  batch[i][0]
            mel = batch[i][1]
            spc = batch[i][2]

            text_input_padded[i,:text.size(0)] = text
            mel_padded[i,  :, :mel.size(1)] = mel
            spc_padded[i,  :, :spc.size(1)] = spc
            emotion_id[i] = batch[i][3][0]
            #make sure the downsampled stop_token_padded have the last eng flag 1.
            stop_token_padded[i, mel.size(1)-self.n_frames_per_step:] = 1


        return text_input_padded, mel_padded, spc_padded, emotion_id, \
                    text_lengths, mel_lengths, stop_token_padded
