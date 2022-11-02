import numpy as np

from ertk.dataset import Dataset, write_filelist


def main():
    rng = np.random.default_rng(54321)

    data = Dataset("../../emotion/datasets/CREMA-D/corpus.yaml", subset="all")
    audio_paths = data.get_audio_paths()
    speakers = rng.permutation(data.speaker_names)

    train_speakers = speakers[:80]
    train_idx = data.get_idx_for_split({"speaker": train_speakers})
    write_filelist(audio_paths[train_idx], "train.txt")
    valid_speakers = speakers[80:90]
    valid_idx = data.get_idx_for_split({"speaker": valid_speakers})
    write_filelist(audio_paths[valid_idx], "valid.txt")

    data.remove_groups("speaker", keep=train_speakers)
    for i, emo in enumerate(data.classes):
        cls_idx = np.flatnonzero(data.y == i)
        emo_paths = audio_paths[cls_idx]
        write_filelist(emo_paths, f"train_{emo}.txt")


if __name__ == "__main__":
    main()
