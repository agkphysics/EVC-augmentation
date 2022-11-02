import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hparams import create_hparams
from model import Parrot
from reader.cv_10lang import TextMelIDCollate
from reader.cv_10lang.extract import extract


class Loader(torch.utils.data.Dataset):
    def __init__(self, file_path_list, mean_std_file):
        self.file_path_list = file_path_list
        self.mel_mean_std = np.float32(np.load(mean_std_file))

    def __getitem__(self, index):
        path = self.file_path_list[index]

        mel, spec = extract(path, trim=False)

        # Normalize audio
        mel = (mel - self.mel_mean_std[0])/ self.mel_mean_std[1]

        # Format for pytorch
        text_input = torch.zeros(100, dtype=torch.int64)
        mel = torch.from_numpy(np.transpose(mel))
        spec = torch.from_numpy(np.transpose(spec))
        speaker_id = torch.LongTensor([0])

        return (text_input, mel, spec, speaker_id)

    def __len__(self):
        return len(self.file_path_list)


def load_model_for_inference(hparams, checkpoint_path):
    model = Parrot(hparams).eval().cuda()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"Loaded model from checkpoint {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='directory to save checkpoints')
    parser.add_argument('--hparams', type=str, required=False, help='comma separated name=value pairs')
    parser.add_argument("--input_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    hparams = create_hparams(args.hparams)

    model = load_model_for_inference(hparams, args.checkpoint)
    model.eval()

    with open(args.input_list) as fid:
        input_list = list(map(str.strip, fid))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Loader(input_list, hparams.mel_mean_std)
    collate_fn = TextMelIDCollate(np.lcm(hparams.n_frames_per_step_encoder, hparams.n_frames_per_step_decoder))
    loader = DataLoader(dataset, num_workers=5, sampler=None, shuffle=False, batch_size=1, drop_last=False, collate_fn=collate_fn)

    mean, std = np.load(hparams.mel_mean_std)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=len(dataset), disable=None)):
            x, _ = model.parse_batch(batch)

            _, post_output, *_ = model.forward(x, input_text=False)

            in_path = Path(input_list[i])
            mel_path = output_dir / f"{in_path.stem}.npy"

            post_output = post_output.data.cpu().numpy()[0]

            mel = post_output * std[:, None] + mean[:, None]
            np.save(mel_path, mel)


if __name__ == "__main__":
    main()
