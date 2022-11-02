import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from hparams import create_hparams
from inference_utils import linear_mel, recover_wav, log_mel
from reader import TextMelIDCollate, extract, ph2id, id2ph
from train import load_model_for_inference
from hifi_gan import Generator


class Loader(torch.utils.data.Dataset):
    def __init__(self, file_path_list, mean_std_file, phones_csv, neutral_id):
        self.file_path_list = file_path_list
        self.mel_mean_std = np.float32(np.load(mean_std_file))
        self.neutral_id = neutral_id

        self.phones = None
        if phones_csv:
            df = pd.read_csv(phones_csv, header=0, index_col=0, dtype=str)
            self.phones = df.squeeze().str.strip()

    def __getitem__(self, index):
        path = self.file_path_list[index]

        mel, spec = extract(path, trim=False, normalise=True)

        if self.phones is not None:
            text_input = self.phones[Path(path).stem].split()
            print(text_input)
            text_input = [ph2id[x] for x in text_input if x in ph2id]
            print(text_input)
        else:
            text_input = [0]

        # Normalize audio
        mel = (mel - self.mel_mean_std[0])/ self.mel_mean_std[1]

        # Format for pytorch
        text_input = torch.LongTensor(text_input)
        mel = torch.from_numpy(np.transpose(mel))
        spec = torch.from_numpy(np.transpose(spec))
        emotion_id = torch.LongTensor([self.neutral_id])

        return (text_input, mel, spec, emotion_id)

    def __len__(self):
        return len(self.file_path_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='Checkpoint to use')
    parser.add_argument('--hparams', type=str, required=False, help='HParams override')
    parser.add_argument('--hifi_gan_path', type=str, help='HiFi-GAN checkpoint to use')
    parser.add_argument("--input_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mel", action="store_true")
    parser.add_argument("--logmel", action="store_true")
    parser.add_argument("--wav", action="store_true")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--neutral", type=int)
    args = parser.parse_args()

    hparams = create_hparams(args.hparams)

    torch.manual_seed(54321)
    torch.cuda.manual_seed(54321)

    model = load_model_for_inference(hparams, args.checkpoint_path)
    model.eval()

    hifi_gan = None
    if args.hifi_gan_path:
        conf_path = Path(args.hifi_gan_path).parent / "config.json"
        with open(conf_path) as fid:
            conf = json.load(fid)
        conf = OmegaConf.create(conf)
        hifi_gan = Generator(conf).cuda()
        checkpoint_dict = torch.load(args.hifi_gan_path, map_location="cuda")
        hifi_gan.load_state_dict(checkpoint_dict["generator"])
        hifi_gan.eval()
        hifi_gan.remove_weight_norm()

    with open(args.input_list) as fid:
        input_list = list(map(str.strip, fid))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Loader(input_list, hparams.mel_mean_std, hparams.phones_csv, args.neutral)
    collate_fn = TextMelIDCollate(np.lcm(hparams.n_frames_per_step_encoder, hparams.n_frames_per_step_decoder))
    loader = DataLoader(dataset, num_workers=1, sampler=None, shuffle=False, batch_size=1, drop_last=False, collate_fn=collate_fn)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=len(dataset), disable=None)):
            x, _ = model.parse_batch(batch)

            for _id, emo in enumerate(hparams.emo_list):
                reference_id = torch.tensor([_id], dtype=torch.int64, device="cuda")
                _, post_output, stop, _, _, _, phids, _, _ = model.inference(x, args.text, reference_id, beam_width=10)

                stop = stop.sigmoid()
                if stop.max().item() < hparams.stop_threshold:
                    print(stop.min().item(), stop.max().item())

                in_path = Path(input_list[i])
                wav_path = output_dir / f"{in_path.stem}_{emo}.wav"
                mel_path = output_dir / f"{in_path.stem}_{emo}.npy"

                post_output = post_output.squeeze().data.cpu().numpy()
                logmel = log_mel(post_output, hparams.mel_mean_std)
                mel = linear_mel(post_output, hparams.mel_mean_std)

                if args.mel:
                    np.save(mel_path, mel)
                if args.logmel:
                    np.save(mel_path, logmel)
                if args.wav:
                    if hifi_gan is not None:
                        logmel = torch.as_tensor(logmel, device="cuda")
                        y_g_hat = hifi_gan(logmel.unsqueeze(0)).squeeze().cpu().numpy()
                        sf.write(wav_path, y_g_hat, 16000, "PCM_16")
                    else:
                        recover_wav(mel, str(wav_path))


if __name__ == "__main__":
    main()
