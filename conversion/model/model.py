from pathlib import Path
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from math import sqrt
from .utils import to_gpu
from .decoder import Decoder
from .layers import SpeakerClassifier, SpeakerEncoder, AudioSeq2seq, TextEncoder,  PostNet, MergeNet
import os


class Parrot(nn.Module):
    def __init__(self, hparams):
        super(Parrot, self).__init__()

        # plus <sos>
        self.embedding = nn.Embedding(
            hparams.n_symbols + 1, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)

        self.sos = hparams.n_symbols
        self.emo_inference_weight = hparams.emo_inference_weight

        self.text_encoder = TextEncoder(hparams)
        self.audio_seq2seq = AudioSeq2seq(hparams)
        self.merge_net = MergeNet(hparams)
        self.speaker_encoder = SpeakerEncoder(hparams)
        self.speaker_encoder.requires_grad_(False)  # Freeze weights
        # self.emotion_encoder = SpeakerEncoder(hparams)
        self.spk_emo_proj = nn.Sequential(
            nn.Linear(2 * hparams.emotion_embedding_dim, hparams.emotion_embedding_dim),
            nn.Tanh(),
            nn.Linear(hparams.emotion_embedding_dim, hparams.emotion_embedding_dim),
            nn.Tanh()
        )
        self.emotion_classifier = SpeakerClassifier(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = PostNet(hparams)
        self._initilize_emb(hparams)

        self.spemb_input = hparams.spemb_input

    def _initilize_emb(self, hparams):
        # a_embedding = np.load(hparams.a_embedding_path).mean(0)
        # b_embedding = np.load(hparams.b_embedding_path).mean(0)
        # c_embedding = np.load(hparams.c_embedding_path).mean(0)
        # d_embedding = np.load(hparams.d_embedding_path).mean(0)
        # e_embedding = np.load(hparams.e_embedding_path).mean(0)

        # self.emo_embedding = nn.Embedding(hparams.n_emotions, hparams.emotion_embedding_dim)
        # for i, emb in zip(range(hparams.n_emotions), [a_embedding, b_embedding, c_embedding, d_embedding, e_embedding]):
        #     self.emo_embedding.weight.data[i] =  torch.FloatTensor(emb)

        self.emo_embedding = nn.Embedding(len(hparams.emo_list), hparams.emotion_embedding_dim)
        for i, emo in enumerate(hparams.emo_list):
            emb_path = Path(hparams.emo_embedding_dir, f"{emo}.npy")
            emb = np.load(emb_path).mean(0)
            self.emo_embedding.weight.data[i] = torch.FloatTensor(emb)

        # self.spk_embedding = nn.Embedding(len(hparams.spk_list), hparams.emotion_embedding_dim)
        # for i, spk in enumerate(hparams.spk_list):
        #     emb_path = Path(hparams.spk_embedding_dir, f"{spk}.npy")
        #     emb = np.load(emb_path).mean(0)
        #     self.spk_embedding.weight.data[i] = torch.FloatTensor(emb)

    def grouped_parameters(self,):
        params_group1 = [p for p in self.embedding.parameters()]
        params_group1.extend([p for p in self.text_encoder.parameters()])
        params_group1.extend([p for p in self.audio_seq2seq.parameters()])

        params_group1.extend([p for p in self.emo_embedding.parameters()])
        params_group1.extend([p for p in self.merge_net.parameters()])
        params_group1.extend([p for p in self.decoder.parameters()])
        params_group1.extend([p for p in self.postnet.parameters()])

        return params_group1, [p for p in self.emotion_classifier.parameters()]

    def parse_batch(self, batch):
        text_input_padded, mel_padded, spc_padded, emotion_id, \
                    text_lengths, mel_lengths, stop_token_padded = batch

        text_input_padded = to_gpu(text_input_padded).long()
        mel_padded = to_gpu(mel_padded).float()
        spc_padded = to_gpu(spc_padded).float()
        emotion_id = to_gpu(emotion_id).long()
        text_lengths = to_gpu(text_lengths).long()
        mel_lengths = to_gpu(mel_lengths).long()
        stop_token_padded = to_gpu(stop_token_padded).float()

        return ((text_input_padded, mel_padded, text_lengths, mel_lengths, emotion_id),
                (text_input_padded, mel_padded, spc_padded,  emotion_id, stop_token_padded))

    def forward(self, inputs, input_text):
        text_input_padded, mel_padded, text_lengths, mel_lengths, emotion_id = inputs

        text_input_embedded = self.embedding(text_input_padded.long()).transpose(1, 2) # -> [B, text_embedding_dim, max_text_len]
        text_hidden = self.text_encoder(text_input_embedded, text_lengths) # -> [B, max_text_len, hidden_dim]

        B = text_input_padded.size(0)
        start_embedding = Variable(text_input_padded.new(B,).fill_(self.sos))
        start_embedding = self.embedding(start_embedding)

        emotion_embedding = self.emo_embedding(emotion_id)
        _, speaker_embedding = self.speaker_encoder(mel_padded, mel_lengths)
        combined_embedding = self.spk_emo_proj(torch.cat([emotion_embedding, speaker_embedding], 1))

        if self.spemb_input:
            T = mel_padded.size(2)
            audio_input = torch.cat((mel_padded,
                combined_embedding.detach().unsqueeze(2).expand(-1, -1, T)), dim=1)
        else:
            audio_input = mel_padded

        #-> [B, text_len+1, hidden_dim] [B, text_len+1, n_symbols] [B, text_len+1, T/r]
        audio_seq2seq_hidden, audio_seq2seq_logit, audio_seq2seq_alignments = self.audio_seq2seq(
                audio_input, mel_lengths, text_input_embedded, start_embedding)
        audio_seq2seq_hidden= audio_seq2seq_hidden[:,:-1, :] # -> [B, text_len, hidden_dim]

        emotion_logit_from_mel_hidden = self.emotion_classifier(audio_seq2seq_hidden) # -> [B, text_len, n_emotions]

        if input_text:
            hidden = self.merge_net(text_hidden, text_lengths)
        else:
            hidden = self.merge_net(audio_seq2seq_hidden, text_lengths)

        L = hidden.size(1)
        hidden = torch.cat([hidden, combined_embedding.unsqueeze(1).expand(-1, L, -1)], -1)

        predicted_mel, predicted_stop, alignments = self.decoder(hidden, mel_padded, text_lengths)

        post_output = self.postnet(predicted_mel)

        outputs = [predicted_mel, post_output, predicted_stop, alignments,
                  text_hidden, audio_seq2seq_hidden, audio_seq2seq_logit, audio_seq2seq_alignments,
                  emotion_logit_from_mel_hidden,
                  text_lengths, mel_lengths]

        return outputs

    def inference(self, inputs, input_text, id_reference, beam_width):

        text_input_padded, mel_padded, text_lengths, mel_lengths, emotion_id = inputs
        text_input_embedded = self.embedding(text_input_padded.long()).transpose(1, 2)
        text_hidden = self.text_encoder.inference(text_input_embedded)

        B = text_input_padded.size(0) # B should be 1
        start_embedding = Variable(text_input_padded.new(B,).fill_(self.sos))
        start_embedding = self.embedding(start_embedding) # [1, embedding_dim]

        #-> [B, text_len+1, hidden_dim] [B, text_len+1, n_symbols] [B, text_len+1, T/r]

        if self.spemb_input:
            T = mel_padded.size(2)
            emotion_embedding = self.emo_embedding(emotion_id) * self.emo_inference_weight
            audio_input = torch.cat((mel_padded, emotion_embedding.unsqueeze(2).expand(-1,-1,T)), dim=1)
        else:
            audio_input = mel_padded

        audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments = self.audio_seq2seq.inference_beam(
                audio_input, start_embedding, self.embedding, beam_width=beam_width)
        audio_seq2seq_hidden= audio_seq2seq_hidden[:,:-1, :] # -> [B, text_len, hidden_dim]

        emotion_embedding = self.emo_embedding(id_reference) * self.emo_inference_weight
        _, speaker_embedding = self.speaker_encoder(mel_padded, mel_lengths)
        combined_embedding = self.spk_emo_proj(torch.cat([emotion_embedding, speaker_embedding], 1))

        if input_text:
            hidden = self.merge_net.inference(text_hidden)
        else:
            hidden = self.merge_net.inference(audio_seq2seq_hidden)

        L = hidden.size(1)
        hidden = torch.cat([hidden, combined_embedding.unsqueeze(1).expand(-1, L, -1)], -1)

        predicted_mel, predicted_stop, alignments = self.decoder.inference(hidden)

        post_output = self.postnet(predicted_mel)

        return (predicted_mel, post_output, predicted_stop, alignments,
            text_hidden, audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments,
            emotion_id)
