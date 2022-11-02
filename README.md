# Emotional voice conversion as data augmentation
This repo is based on and contains code from the following other
projects:
- [HiFi-GAN](https://github.com/jik876/hifi-gan/)
- [seq2seq EVC](https://github.com/KunZhou9646/seq2seq-EVC)
- [ERTK](https://github.com/Strong-AI-Lab/emotion)


## Data
The datasets we use are IEMOCAP, MSP-IMPROV, EmoV-DB, CREMA-D and ESD.
Please download these datasets and use ERTK to preprocess the data,
extract features, and extract phone transcriptions.

### Data preprocessing
Firstly, download the relevant datasets and preprocess using the ERTK
scripts. For example, for CREMA-D:
```
cd emotion/datasets/CREMA-D
python process.py /path/to/CREMA-D
```

#### Phonemization
Extract phone transcriptions of the transcripts, e.g.:
```
cd emotion/datasets/CREMA-D
ertk-dataset process \
    --features phonemize \
    --batch_size -1 \
    transcript.csv \
    phones_ipa.csv \
    language=en-us \
    backend=espeak \
    language_switch=remove-flags
```
For ESD we need to do Chinese and English separately:
```
cd emotion/datasets/ESD
ertk-dataset process \
    --features phonemize \
    --batch_size -1 \
    transcripts_en.csv \
    phones_ipa.csv \
    language=en-us \
    backend=espeak \
    language_switch=remove-flags
ertk-dataset process \
    --features phonemize \
    --batch_size -1 \
    transcripts_zh.csv \
    phones_ipa.csv \
    language=cmn \
    backend=espeak \
    language_switch=remove-flags
```

For MSP-IMPROV, since no official transcripts are given, and a number of
clips have arbitrary text content, we use a state-of-the-art speech
recogniser to generate transcripts.
```
cd emotion/datasets/MSP-IMPROV
ertk-dataset process \
    --features huggingface \
    files_all.txt \
    transcript_w2v.csv \
    model=facebook/wav2vec2-large-960h-lv60-self
ertk-dataset process \
    --features phonemize \
    --batch_size -1 \
    transcript_w2v.csv \
    phones_ipa.csv \
    language=en-us \
    backend=espeak \
    language_switch=remove-flags
```

#### Feature extraction
Then extract features for real data from pretrained and fine-tuned
embeddings:
```
ertk-dataset process \
    --features fairseq \
    --corpus CMU-MOSEI \
    --sample_rate 16000 \
    datasets/CMU-MOSEI/files_all.txt \
    features/CMU-MOSEI/wav2vec_c_mean.nc \
    model_type=wav2vec \
    checkpoint=/path/to/wav2vec_large.pt \
    layer=context \
    aggregate=MEAN
```
```
ertk-dataset process \
    --features huggingface \
    --corpus CMU-MOSEI \
    --sample_rate 16000 \
    datasets/CMU-MOSEI/files_all.txt \
    features/CMU-MOSEI/wav2vec_audeering_ft_c_mean.nc \
    model=audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim \
    task=EMBEDDINGS \
    layer=context \
    agg=MEAN
```


## Emotional voice conversion
The `gen_evc_data.py` scripts in the `data/*` directories will generate
train and validation subsets for training the EVC models.

### Pre-training EVC models
The `pre-train` directory is modelled similarly to
[seq2seq EVC](https://github.com/KunZhou9646/seq2seq-EVC) so also check
there for additional details.

#### Common Voice
We use a 10-language subset of
[Common Voice](https://commonvoice.mozilla.org/en/datasets) 10.0
consisting of Chinese, English, Arabic, Greek, Bangla, Farsi, Italian,
Portuguese, Urdu and Estonian.

The `gen_evc_data.py` file in `data/CommonVoice/` will generate data for
EVC pretraining:
```
python gen_evc_data.py /path/to/CommonVoice
```
This will generate train/validation subsets, as well as phone
transcriptions and speaker information for each clip.

#### Mel standardisation
You need to generate the mel spectrogram standardisation file using
`mel_mean_std.py` like so:
```
python mel_mean_std.py data/CommonVoice/train.txt data/CommonVoice
```
This will generate the `mel_mean_std.npy` file containing mean and
standard deviation of each mel-band over the whole training data.

#### Training
The main training script is `train.py`. Hyperparameters are stored in
`hparams.py` and can be modified either there or on the command line.
Data is read with the `reader.py` file in `reader/cv_10lang/reader.py`,
and spectrograms are generated as needed in memory.

You can train with the following command, adjust as per your setup:
```
python train.py \
    --output_directory out_cv_10lang \
    --hparams distributed_run=True,batch_size=64,training_list=data/CommonVoice/train.txt,validation_list=data/CommonVoice/dev.txt,mel_mean_std=data/CommonVoice/mel_mean_std.npy,phones_csv=data/CommonVoice/phones_ipa.csv,speaker_csv=data/CommonVoice/speaker.csv,n_speakers=1967,n_symbols=315 \
    --n_gpus 4
```

We also provide a model weights for a pretrained model
[here](https://drive.google.com/file/d/1ja824x5j8kzBy2JFurHSYOCctY1rUWRm/view?usp=sharing).

### Fine-tuning on each dataset
The `conversion` directory has a similar structure to the `pre-train`
directory. The main differences are that the model weights are
initialised from the pretrained model, and the model structure is
slightly different, as mentioned in the paper.

Use the `gen_embedding.py` file to generate initial emotion embeddings
from the pretrained model:
```
cd conversion
for emo in anger disgust fear happiness neutral sadness; do \
    python gen_embedding.py \
        --checkpoint_path ../pre-train/out_cv_10lang/logdir/checkpoint_84000 \
        --hparams mel_mean_std=../data/CREMA-D/mel_mean_std.npy,pretrain_n_speakers=1967,n_symbols=315 \
        --input ../data/CREMA-D/train_${emo}.txt \
        --output embeddings/CREMA-D/cv_10lang/${emo}.npy; \
done
```

The fine-tuning commands are given in `run_train.sh`.

### HiFi-GAN training
The `hifi-gan` directory contains scripts to pretrain and finetune a
HiFi-GAN model. We pretrain the model on audio files from the common
voice data, using the same train/validation splits.

#### Pretraining
```
cd hifi-gan
python train.py \
    --input_training_file data/CommonVoice/train.txt \
    --input_validation_file data/CommonVoice/dev.txt \
    --config config_v1_16k_0.05_0.0125.json \
    --checkpoint_path cp/v1_cv_10lang \
    --checkpoint_interval 2000 \
    --validation_interval 2000 \
    --summary_interval 200
```
We provide a HiFi-GAN model pretrained on the Common Voice data
[here](https://drive.google.com/drive/folders/121e6UgV1qtKGTbdEoGmQhp4BdAQqJUrB?usp=sharing)

#### Finetuning
First generate forward outputs from the EVC model using
`gen_fwd_mels.py`:
```
cd pre-train
python gen_fwd_mels.py \
    --checkpoint out_cv_10lang/logdir/checkpoint_84000 \
    --output_dir out_cv_10lang/fwd_mels \
    --input_list ../data/CommonVoice/train.txt \
    --hparams training_list=../data/CommonVoice/train.txt,validation_list=../data/CommonVoice/dev.txt,mel_mean_std=../data/CommonVoice/mel_mean_std.npy,phones_csv=../data/CommonVoice/phones_ipa.csv,speaker_csv=../data/CommonVoice/speaker.csv,n_speakers=1967,n_symbols=315
python gen_fwd_mels.py \
    --checkpoint out_cv_10lang/logdir/checkpoint_84000 \
    --output_dir out_cv_10lang/fwd_mels \
    --input_list ../data/CommonVoice/dev.txt \
    --hparams training_list=../data/CommonVoice/train.txt,validation_list=../data/CommonVoice/dev.txt,mel_mean_std=../data/CommonVoice/mel_mean_std.npy,phones_csv=../data/CommonVoice/phones_ipa.csv,speaker_csv=../data/CommonVoice/speaker.csv,n_speakers=1967,n_symbols=315
```

To finetune HiFi-GAN, copy the pretrained checkpoint to a new directory:
```
cd hifi-gan
mkdir cp/v1_cv_10lang_ft_cv_10lang
cp cp/v1_cv_10lang/{g,do}_00152000 cp/v1_cv_10lang_ft_cv_10lang/
```

Then use the finetuning version of the script:
```
cd hifi-gan
python train.py \
    --fine_tuning True \
    --input_training_file ../data/CommonVoice/train_100k_local.txt \
    --input_validation_file ../data/CommonVoice/dev.txt \
    --input_mels_dir ../pre-train/out_cv_10lang/fwd_mels \
    --config config_v1_16k_0.05_0.0125.json \
    --checkpoint_path cp/v1_cv_10lang_ft_cv_10lang/ \
    --checkpoint_interval 2000 \
    --validation_interval 1000 \
    --summary_interval 200
```

We provide a HiFi-GAN model finetuned on model outputs from the
Common Voice data [here](https://drive.google.com/file/d/17HeQG0kG72FUDGuDvPmGUO4vd8mASqDS/view?usp=share_link)

### Emotion conversion
The fine-tuning process gets the model to learn to disentangle speaker,
emotion and linguistic information, so can impute emotional expressions
into some given speech.

Use the `convert_all.py` file to convert a given set of speech files
into all trained emotion categories:
```
cd conversion
python convert_all.py \
    --checkpoint_path out_ft_IEMOCAP_en_4class/logdir/checkpoint_13651 \
    --input_list emotion/datasets/MSP-IMPROV/files_neutral.txt \
    --output_dir augmentation/datasets/MSP-IMPROV_aug/IEMOCAP_evc/hifi-gan_v1_ft_mel_vocoded/ \
    --wav \
    --hifi_gan_path hifi-gan/cp/v1_cv_10lang_ft_cv_10lang/g_00496000 \
    --hparams \""emo_list=[anger,happiness,neutral,sadness]"\",emo_embedding_dir=embeddings/IEMOCAP/,mel_mean_std=../data/IEMOCAP/mel_mean_std.npy,pretrain_n_speakers=1967,n_symbols=315 \
    --neutral 2
```


## Augmentation experiments
After converting the data you should extract features similarly to the
real data:
```
cd augmentation/datasets/MSP-IMPROV_aug/MSP-IMPROV_evc
find hifi-gan_v1_ft_mel_vocoded -type f | sort > files.txt
ertk-dataset process \
    --features fairseq \
    --corpus CMU-MOSEI \
    --sample_rate 16000 \
    files.txt \
    wav2vec_c_mean.nc \
    model_type=wav2vec \
    checkpoint=/path/to/wav2vec_large.pt \
    layer=context \
    aggregate=MEAN
```

The augmented datasets can be aligned using `align_aug_annots.py`:
```
cd augmentation/datasets
python align_aug_annots.py \
    IEMOCAP_aug \
    --original ../../emotion/datasets/IEMOCAP/corpus.yaml \
    --annot speaker \
    --annot session
```
This will generate a `corpus.yaml` file for each augmented dataset and
create the appropriate metadata, including subsets for each EVC model
and specified annotations.

The experiments can be generated using the `gen_evc_runs.sh` script for
each feature set:
```
bash gen_evn_runs.sh > jobs.txt
```

You can run multiple experiments in parallel using
```
ertk-util parallel_jobs jobs.txt --cpus 24 --failed failed.txt
```
