import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hparams import create_hparams
from reader import TextMelIDCollate, extract
from train import load_model_for_inference


class Loader(torch.utils.data.Dataset):
    def __init__(self, list_file, mean_std_file):
        with open(list_file) as f:
            self.file_path_list = list(map(str.strip, f))
        self.mel_mean_std = np.float32(np.load(mean_std_file))

    def get_text_mel_id_pair(self, path):
        mel, spec = extract(path, trim=True)

        # Normalize audio
        mel = (mel - self.mel_mean_std[0])/ self.mel_mean_std[1]

        # Format for pytorch
        text_input = torch.LongTensor([0])
        mel = torch.from_numpy(np.transpose(mel))
        spec = torch.from_numpy(np.transpose(spec))
        emotion_id = torch.LongTensor([0])

        return (text_input, mel, spec, emotion_id)

    def __getitem__(self, index):
        return self.get_text_mel_id_pair(self.file_path_list[index])

    def __len__(self):
        return len(self.file_path_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--input', type=Path,
                        required=True, help="File containing audio filenames to process.")
    parser.add_argument('--output', type=Path,
                        required=True, help="File to output embeddings.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    hparams = create_hparams(args.hparams)
    model = load_model_for_inference(hparams, checkpoint_path)
    model.eval()

    dataset = Loader(args.input, hparams.mel_mean_std)
    collate_fn = TextMelIDCollate(np.lcm(hparams.n_frames_per_step_encoder, hparams.n_frames_per_step_decoder))
    data_loader = DataLoader(dataset, num_workers=1, shuffle=False, batch_size=1,
                             collate_fn=collate_fn)
    with torch.no_grad():
        embeddings = []
        for batch in tqdm(data_loader, total=len(dataset), disable=None):
            x, _ = model.parse_batch(batch)
            _, mel_padded, _, _, _ = x
            _, embedding = model.speaker_encoder.inference(mel_padded)
            embeddings.append(embedding.cpu().numpy())
        embeddings = np.vstack(embeddings)

    print(embeddings.shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
