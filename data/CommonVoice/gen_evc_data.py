import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ertk.dataset import write_filelist
from ertk.preprocessing.phonemize import PhonemizeConfig, Phonemizer

base_dir = Path(sys.argv[1])
rng = np.random.default_rng(54321)

train_paths = []
dev_paths = []
all_phones = []
all_speakers = []
for lang_dir in base_dir.glob("*"):
    if not lang_dir.is_dir():
        continue

    print(lang_dir)
    df = pd.read_csv(
        lang_dir / "validated.tsv", sep="\t", header=0, low_memory=False, quoting=3
    )
    df.index = df["path"].str[:-4]
    df = df.sort_index()
    df["path"] = str(lang_dir / "clips") + "/" + df["path"]
    df["speaker"] = df["client_id"].astype("category")

    espeak_lang = lang_dir.stem
    if espeak_lang == "en":
        espeak_lang = "en-us"
    elif espeak_lang == "zh-CN":
        espeak_lang = "cmn"
    phonemizer = Phonemizer(
        PhonemizeConfig(
            backend="espeak", language=espeak_lang, language_switch="remove-flags"
        )
    )
    phones = np.concatenate(phonemizer.process_batch(df["sentence"].to_numpy()))
    phones = pd.Series(phones, index=df.index)
    phones = phones.str.strip()

    phone_len = phones.map(lambda x: len(x.split()))

    use_speakers = df["speaker"].value_counts().head(200).index
    assert len(use_speakers) <= 200
    df = df.loc[(df["speaker"].isin(use_speakers) & (phone_len <= 100))]

    train_names: List[str] = []
    dev_names: List[str] = []
    for spk in use_speakers:
        spk_names = rng.permuted(df.index[df["speaker"] == spk])
        n_clips = len(spk_names)
        n_train = int(0.99 * n_clips)
        train_names.extend(spk_names[:n_train])
        dev_names.extend(spk_names[n_train:n_clips])

    print(len(train_names), len(dev_names))

    all_phones.append(phones[train_names + dev_names])
    all_speakers.append(df.loc[train_names + dev_names, "speaker"])
    train_paths.extend(df.loc[train_names, "path"])
    dev_paths.extend(df.loc[dev_names, "path"])

write_filelist(train_paths, "train.txt")
write_filelist(dev_paths, "dev.txt")
pd.concat(all_phones).to_csv("phones_ipa.csv")
pd.concat(all_speakers).to_csv("speaker.csv")
