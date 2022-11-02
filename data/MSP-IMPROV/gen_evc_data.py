import numpy as np

from ertk.dataset import Dataset, write_filelist, read_annotations


def main():
    phones = read_annotations("./phones_us.csv", dtype=str).str.strip()
    phone_len = phones.map(lambda x: len(x.split()), na_action="ignore")
    invalid_names = phones[phones.isna() | (phone_len > 100)].index

    train_data = Dataset("./corpus.yaml", subset="4class")
    train_data.remove_groups("session", keep=["01", "02", "03", "04"])
    drop_names = invalid_names.intersection(train_data.names)
    if len(drop_names) > 0:
        train_data.remove_instances(drop=drop_names)

    audio_paths = train_data.get_audio_paths()
    write_filelist(audio_paths, "evc/train.txt")
    for i, emo in enumerate(train_data.classes):
        cls_idx = np.flatnonzero(train_data.y == i)
        emo_paths = audio_paths[cls_idx]
        write_filelist(emo_paths, f"evc/train_{emo}.txt")

    valid_data = Dataset("./corpus.yaml", subset="4class")
    valid_data.remove_groups("session", keep=["05"])
    drop_names = invalid_names.intersection(valid_data.names)
    if len(drop_names) > 0:
        valid_data.remove_instances(drop=drop_names)
    write_filelist(valid_data.get_audio_paths(), "evc/valid.txt")


if __name__ == "__main__":
    main()
