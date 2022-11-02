#!/bin/bash

for lang in en zh; do
    for features in wav2vec_c_mean wav2vec2_audeering_ft_c_mean; do
        echo ertk-cli exp2 configs/within/ESD_train.yaml model.type=sk/lr model.config="\\\${cwdpath:../emotion/conf/clf/sk/lr/default.yaml}" data.features=$features data.remove_groups.language.keep=[$lang] training.n_jobs=1 results=augmentation/results/within/esd_train/$features/noaug/$lang.csv
        for aug in ESD_en_evc ESD_zh_evc; do
            echo ertk-cli exp2 configs/within/cv_ESD_train_aug.yaml model.type=sk/lr model.config="\\\${cwdpath:../emotion/conf/clf/sk/lr/default.yaml}" data.features=datasets/ESD_aug/$aug/$features.nc data.datasets.ESD.subset=$aug data.remove_groups.language.keep=[$lang] training.n_jobs=1 results=augmentation/results/within/esd_train/$features/$aug/$lang.csv
        done
    done
done

for features in wav2vec_c_mean wav2vec2_audeering_ft_c_mean; do
    echo ertk-cli exp2 configs/within/IEMOCAP.yaml model.type=sk/lr model.config="\\\${cwdpath:../emotion/conf/clf/sk/lr/default.yaml}" data.features=$features training.n_jobs=1 results=augmentation/results/within/cross_corpus_induced_4class/$features/noaug/IEMOCAP.csv
    echo ertk-cli exp2 configs/within/MSP-IMPROV.yaml model.type=sk/lr model.config="\\\${cwdpath:../emotion/conf/clf/sk/lr/default.yaml}" data.features=$features training.n_jobs=1 results=augmentation/results/within/cross_corpus_induced_4class/$features/noaug/MSP-IMPROV.csv
    for aug in IEMOCAP_evc MSP-IMPROV_evc; do
        echo ertk-cli exp2 configs/within/cv_IEMOCAP_aug.yaml model.type=sk/lr model.config="\\\${cwdpath:../emotion/conf/clf/sk/lr/default.yaml}" data.features=datasets/IEMOCAP_aug/$aug/$features.nc data.datasets.IEMOCAP.subset=$aug training.n_jobs=1 results=augmentation/results/within/cross_corpus_induced_4class/$features/$aug/IEMOCAP.csv
        echo ertk-cli exp2 configs/within/cv_MSP-IMPROV_aug.yaml model.type=sk/lr model.config="\\\${cwdpath:../emotion/conf/clf/sk/lr/default.yaml}" data.features=datasets/MSP-IMPROV_aug/$aug/$features.nc data.datasets.MSP-IMPROV.subset=$aug training.n_jobs=1 results=augmentation/results/within/cross_corpus_induced_4class/$features/$aug/MSP-IMPROV.csv
    done
done

for features in wav2vec_c_mean wav2vec2_audeering_ft_c_mean; do
    echo ertk-cli exp2 configs/within/EmoV-DB.yaml model.type=sk/lr model.config="\\\${cwdpath:../emotion/conf/clf/sk/lr/default.yaml}" data.features=$features training.n_jobs=1 results=augmentation/results/within/cross_corpus_explicit_4class/$features/noaug/EmoV-DB.csv
    echo ertk-cli exp2 configs/within/CREMA-D.yaml model.type=sk/lr model.config="\\\${cwdpath:../emotion/conf/clf/sk/lr/default.yaml}" data.features=$features training.n_jobs=1 results=augmentation/results/within/cross_corpus_explicit_4class/$features/noaug/CREMA-D.csv
    for aug in CREMA-D_evc EmoV-DB_evc; do
        echo ertk-cli exp2 configs/within/cv_EmoV-DB_aug.yaml model.type=sk/lr model.config="\\\${cwdpath:../emotion/conf/clf/sk/lr/default.yaml}" data.features=datasets/EmoV-DB_aug/$aug/$features.nc data.datasets.EmoV-DB.subset=$aug training.n_jobs=1 results=augmentation/results/within/cross_corpus_explicit_4class/$features/$aug/EmoV-DB.csv
        echo ertk-cli exp2 configs/within/cv_CREMA-D_aug.yaml model.type=sk/lr model.config="\\\${cwdpath:../emotion/conf/clf/sk/lr/default.yaml}" data.features=datasets/CREMA-D_aug/$aug/$features.nc data.datasets.CREMA-D.subset=$aug training.n_jobs=1 results=augmentation/results/within/cross_corpus_explicit_4class/$features/$aug/CREMA-D.csv
    done
done
