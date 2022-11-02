import numpy as np

from ertk.dataset import Dataset, write_filelist, read_annotations


def main():
    rng = np.random.default_rng(54321)

    phones = read_annotations("../../emotion/datasets/EmoV-DB/phones_ipa.csv", dtype=str).str.strip()
    phone_len = phones.map(lambda x: len(x.split()), na_action="ignore")
    invalid_phones = phones[phones.isna() | (phone_len > 100)].index

    data = Dataset("../../emotion/datasets/EmoV-DB/corpus.yaml", subset="all")
    invalid_phones = invalid_phones.intersection(data.names)
    if len(invalid_phones) > 0:
        data.remove_instances(drop=invalid_phones)
    audio_paths = data.get_audio_paths()

    speakers = data.annotations["speaker"].cat.remove_unused_categories()
    use_speakers = ["bea", "jenie", "sam"]

    train_names = []
    valid_names = []
    for spk in use_speakers:
        spk_names = rng.permuted(speakers.index[speakers == spk])
        n_train = int(len(spk_names) * 0.9)
        n_valid = int(len(spk_names) * 0.1)
        train_names.extend(spk_names[:n_train])
        valid_names.extend(spk_names[n_train : n_train + n_valid])
    write_filelist(audio_paths[train_names], "train.txt")
    write_filelist(audio_paths[valid_names], "valid.txt")

    data.remove_instances(keep=train_names)
    audio_paths = data.get_audio_paths()
    for i, emo in enumerate(data.classes):
        emo_paths = audio_paths[data.y == i]
        write_filelist(emo_paths, f"train_{emo}.txt")


if __name__ == "__main__":
    main()
