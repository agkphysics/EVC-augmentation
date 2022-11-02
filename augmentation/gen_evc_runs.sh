#!/bin/bash

lr_model_args="model.type=sk/lr model.args_path=../emotion/conf/clf/sk/lr/default.yaml model.param_grid_path="
model_args="$lr_model_args"
clf=lr
reps=20
# features=wav2vec_c_mean
features=wav2vec2_audeering_ft_c_mean
feature_args="data.features=${features}"
results_dir=results/${clf}_rep${reps}_max_train/${features}
config_dir=configs/evc

p_reals=(0 0.1 0.25 0.5 0.75 0.9 1)
max_trains=(0.25 0.5 0.75 1)

function exp_add_aug() {
    local eval=$1
    local exp=$2
    local group=$3
    local src=$4
    local tgt=$5
    local aug=$6
    local conf=${exp}_aug
    local aug_data_args=$7

    local config=${config_dir}/${conf}.yaml
    local results=${results_dir}/${exp}/${aug}
    for f in "${p_reals[@]}"; do
        local eval_args
        if [ "$eval" == "n" ]; then
            eval_args="eval.train.groups.${group}.keep=[${src}] eval.test.groups.${group}.keep=[${tgt}]"
        else
            eval_args="--eval ${tgt}"
        fi
        local cmd="echo python evc_aug.py --p_real 1 --p_fake $f --reps $reps --config $config $aug_data_args $eval_args training.n_jobs=1 $model_args $feature_args"
        $cmd --results "$results/${src}_${tgt}_aug_target_${f}.csv" aug_data.remove_groups.${group}.keep=[${tgt}]
        $cmd --results "$results/${src}_${tgt}_aug_source_${f}.csv" aug_data.remove_groups.${group}.keep=[${src}]
        $cmd --results "$results/${src}_${tgt}_aug_both_${f}.csv" aug_data.remove_groups.${group}.keep=[${src},${tgt}]
    done
}

function exp_fixed_aug() {
    local eval=$1
    local exp=$2
    local group=$3
    local src=$4
    local tgt=$5
    local aug=$6
    local conf=${exp}_aug
    local aug_data_args=$7
    local use_max_train=$8
    if [ "$use_max_train" = "" ]; then
        local max_trains=("${max_trains[@]}")
    else
        local max_trains=("$use_max_train")
    fi

    local config=${config_dir}/${conf}.yaml
    local results=${results_dir}/${exp}/${aug}_const
    for r in "${p_reals[@]}"; do
        for max_train in "${max_trains[@]}"; do
            local eval_args
            if [ "$eval" == "n" ]; then
                eval_args="eval.train.groups.${group}.keep=[${src}] eval.test.groups.${group}.keep=[${tgt}]"
            else
                eval_args="--eval ${tgt}"
            fi
            local cmd="echo python evc_aug.py --p_real $r --p_fake -1 --max_train $max_train --reps $reps --config $config $aug_data_args $eval_args training.n_jobs=1 $model_args $feature_args"
            $cmd --results "$results/${src}_${tgt}_aug_target_${r}_${max_train}.csv" aug_data.remove_groups.${group}.keep=[${tgt}]
            $cmd --results "$results/${src}_${tgt}_aug_source_${r}_${max_train}.csv" aug_data.remove_groups.${group}.keep=[${src}]
            $cmd --results "$results/${src}_${tgt}_aug_both_${r}_${max_train}.csv" aug_data.remove_groups.${group}.keep=[${src},${tgt}]
        done
    done
}

function exp_fixed_neutral_aug() {
    local eval=$1
    local exp=$2
    local group=$3
    local src=$4
    local tgt=$5
    local aug=$6
    local conf=${exp}_aug
    local aug_data_args=$7
    if [ $# -eq 8 ]; then
        local max_train=$8
    else
        local max_train=1
    fi

    local config=${config_dir}/${conf}.yaml
    local results=${results_dir}/${exp}/${aug}_const_neutral
    for r in "${p_reals[@]}"; do
        if [ "$r" = "0" ]; then continue; fi
        local eval_args
        if [ "$eval" == "n" ]; then
            eval_args="eval.train.groups.${group}.keep=[${src}] eval.test.groups.${group}.keep=[${tgt}]"
        else
            eval_args="--eval ${tgt}"
        fi
        local cmd="echo python evc_aug.py --p_real $r --p_fake -1 --max_train $max_train --reps $reps --config $config $aug_data_args aug_data.remove_groups.label.keep=[neutral] $eval_args training.n_jobs=1 $model_args $feature_args"
        $cmd --results "$results/${src}_${tgt}_augneutral_target_${r}_${max_train}.csv" aug_data.remove_groups.${group}.keep=[${tgt}]
        $cmd --results "$results/${src}_${tgt}_augneutral_source_${r}_${max_train}.csv" aug_data.remove_groups.${group}.keep=[${src}]
        $cmd --results "$results/${src}_${tgt}_augneutral_both_${r}_${max_train}.csv" aug_data.remove_groups.${group}.keep=[${src},${tgt}]
        local cmd="echo python evc_aug.py --p_real $r --p_fake -1 --max_train $max_train --reps $reps --config $config $aug_data_args $eval_args training.n_jobs=1 $model_args $feature_args"
        $cmd --results "$results/${src}_${tgt}_augall_target_${r}_${max_train}.csv" aug_data.remove_groups.${group}.keep=[${tgt}]
        $cmd --results "$results/${src}_${tgt}_augall_source_${r}_${max_train}.csv" aug_data.remove_groups.${group}.keep=[${src}]
        $cmd --results "$results/${src}_${tgt}_augall_both_${r}_${max_train}.csv" aug_data.remove_groups.${group}.keep=[${src},${tgt}]
    done
}

function exp_within_aug() {
    local eval=$1
    local exp=$2
    local group=$3
    local src=$4
    local aug=$5
    local conf=${exp}_aug
    local aug_data_args=$6

    local config=${config_dir}/${conf}.yaml
    local results=${results_dir}/${exp}/${aug}_const
    for r in "${p_reals[@]}"; do
        for max_train in "${max_trains[@]}"; do
            local cmd="echo python evc_aug.py --max_train $max_train --p_real $r --p_fake -1 --reps $reps --config $config $aug_data_args training.n_jobs=1 $model_args $feature_args"
            if [ "$eval" == "n" ]; then
                $cmd --results "$results/${src}_${src}_aug_within_${r}_${max_train}.csv" eval.train.groups.${group}.keep=[${src}] eval.test.groups.${group}.keep=[${src}] aug_data.remove_groups.${group}.keep=[${src}]
            else
                $cmd --results "$results/${src}_${src}_aug_within_${r}_${max_train}.csv" --eval "${src}" aug_data.remove_groups.${group}.keep=[${src}]
            fi
        done
    done
}

function exp_manual() {
    local exp=$1
    local src=$2
    local tgt=$3
    local aug=$4
    local conf=${exp}_aug
    local aug_data_args=$5

    local config=${config_dir}/${conf}.yaml
    local results=${results_dir}/${exp}/${aug}_const
    for r in "${p_reals[@]}"; do
        for max_train in "${max_trains[@]}"; do
            local cmd="echo python evc_aug.py --max_train $max_train --p_real $r --p_fake -1 --reps $reps --config $config $aug_data_args training.n_jobs=1 $model_args $feature_args"
            $cmd --results "$results/${src}_${tgt}_aug_${r}_${max_train}.csv"
        done
    done
}

function exp_real() {
    local eval=$1
    local exp=$2
    local group=$3
    local src=$4
    local tgt=$5
    local conf=${exp}

    local config=${config_dir}/${conf}.yaml
    local results=${results_dir}/${exp}/noaug_const
    for r in "${p_reals[@]}"; do
        for max_train in "${max_trains[@]}"; do
            local cmd="echo python evc_aug.py --max_train $max_train --p_real $r --p_fake -1 --reps $reps --config $config training.n_jobs=1 $model_args $feature_args"
            if [ "$eval" == "n" ]; then
                $cmd --results "$results/${src}_${tgt}_real_target_${r}_${max_train}.csv" eval.train.groups.${group}.keep=[${src}] eval.test.groups.${group}.keep=[${tgt}] aug_data.remove_groups.${group}.keep=[${tgt}]
            else
                $cmd --results "$results/${src}_${tgt}_real_target_${r}_${max_train}.csv" --eval "${tgt}" aug_data.remove_groups.${group}.keep=[${tgt}]
            fi
        done
    done
}

function exp_real_neutral() {
    local eval=$1
    local exp=$2
    local group=$3
    local src=$4
    local tgt=$5
    local conf=${exp}
    if [ $# -eq 6 ]; then
        local max_train=$6
    else
        local max_train=1
    fi

    local config=${config_dir}/${conf}.yaml
    local results=${results_dir}/${exp}/noaug_const_neutral
    for r in "${p_reals[@]}"; do
        if [ "$r" = "0" ]; then continue; fi
        local eval_args
        if [ "$eval" == "n" ]; then
            eval_args="eval.train.groups.${group}.keep=[${src}] eval.test.groups.${group}.keep=[${tgt}]"
        else
            eval_args="--eval ${tgt}"
        fi
        echo python evc_aug.py --p_real $r --p_fake -1 --max_train $max_train --reps $reps --config $config aug_data.remove_groups.label.keep=[neutral] $eval_args training.n_jobs=1 $model_args $feature_args --results "$results/${src}_${tgt}_realneutral_target_${r}_${max_train}.csv" aug_data.remove_groups.${group}.keep=[${tgt}]
        echo python evc_aug.py --p_real $r --p_fake -1 --max_train $max_train --reps $reps --config $config $eval_args training.n_jobs=1 $model_args $feature_args --results "$results/${src}_${tgt}_realall_target_${r}_${max_train}.csv" aug_data.remove_groups.${group}.keep=[${tgt}]
    done
}


## ESD test
exp_fixed_aug n esd_test language en zh ESD_en_evc "aug_data.datasets.aug.subset=ESD_en_evc aug_data.features=datasets/ESD_aug/ESD_en_evc/\\\${data.features}.nc"
exp_fixed_neutral_aug n esd_test language en zh ESD_en_evc "aug_data.datasets.aug.subset=ESD_en_evc aug_data.features=datasets/ESD_aug/ESD_en_evc/\\\${data.features}.nc" 3000
exp_within_aug n esd_test language en ESD_en_evc "aug_data.datasets.aug.subset=ESD_en_evc aug_data.features=datasets/ESD_aug/ESD_en_evc/\\\${data.features}.nc"
exp_add_aug n esd_test language en zh ESD_en_evc "aug_data.datasets.aug.subset=ESD_en_evc aug_data.features=datasets/ESD_aug/ESD_en_evc/\\\${data.features}.nc"

exp_fixed_aug n esd_test language zh en ESD_en_evc "aug_data.datasets.aug.subset=ESD_en_evc aug_data.features=datasets/ESD_aug/ESD_en_evc/\\\${data.features}.nc"
exp_fixed_neutral_aug n esd_test language zh en ESD_en_evc "aug_data.datasets.aug.subset=ESD_en_evc aug_data.features=datasets/ESD_aug/ESD_en_evc/\\\${data.features}.nc" 3000
exp_within_aug n esd_test language zh ESD_en_evc "aug_data.datasets.aug.subset=ESD_en_evc aug_data.features=datasets/ESD_aug/ESD_en_evc/\\\${data.features}.nc"
exp_add_aug n esd_test language zh en ESD_en_evc "aug_data.datasets.aug.subset=ESD_en_evc aug_data.features=datasets/ESD_aug/ESD_en_evc/\\\${data.features}.nc"

exp_real n esd_test language en zh
exp_real n esd_test language zh en
exp_real_neutral n esd_test language en zh 3000
exp_real_neutral n esd_test language zh en 3000


## Cross corpus explicit 4class (CREMA-D, EmoV-DB) test sets
exp_fixed_aug y cross_corpus_explicit_4class_test corpus CREMA-D EmoV-DB CREMA-D_evc "aug_data.datasets.CREMA-D.subset=CREMA-D_evc aug_data.datasets.EmoV-DB.subset=CREMA-D_evc"
exp_fixed_neutral_aug y cross_corpus_explicit_4class_test corpus CREMA-D EmoV-DB CREMA-D_evc "aug_data.datasets.CREMA-D.subset=CREMA-D_evc aug_data.datasets.EmoV-DB.subset=CREMA-D_evc" 967
exp_add_aug y cross_corpus_explicit_4class_test corpus CREMA-D EmoV-DB CREMA-D_evc "aug_data.datasets.CREMA-D.subset=CREMA-D_evc aug_data.datasets.EmoV-DB.subset=CREMA-D_evc"

exp_fixed_aug y cross_corpus_explicit_4class_test corpus EmoV-DB CREMA-D EmoV-DB_evc "aug_data.datasets.CREMA-D.subset=EmoV-DB_evc aug_data.datasets.EmoV-DB.subset=EmoV-DB_evc"
exp_fixed_neutral_aug y cross_corpus_explicit_4class_test corpus EmoV-DB CREMA-D EmoV-DB_evc "aug_data.datasets.CREMA-D.subset=EmoV-DB_evc aug_data.datasets.EmoV-DB.subset=EmoV-DB_evc" 1134
exp_add_aug y cross_corpus_explicit_4class_test corpus EmoV-DB CREMA-D EmoV-DB_evc "aug_data.datasets.CREMA-D.subset=EmoV-DB_evc aug_data.datasets.EmoV-DB.subset=EmoV-DB_evc"

exp_real y cross_corpus_explicit_4class_test corpus CREMA-D EmoV-DB
exp_real y cross_corpus_explicit_4class_test corpus EmoV-DB CREMA-D
exp_real_neutral y cross_corpus_explicit_4class_test corpus CREMA-D EmoV-DB 967
exp_real_neutral y cross_corpus_explicit_4class_test corpus EmoV-DB CREMA-D 1134


## Cross corpus induced 4class (MSP-IMPROV, IEMOCAP) test sets
exp_fixed_aug y cross_corpus_induced_4class_test corpus IEMOCAP MSP-IMPROV IEMOCAP_evc "aug_data.datasets.IEMOCAP.subset=IEMOCAP_evc aug_data.datasets.MSP-IMPROV.subset=IEMOCAP_evc"
exp_fixed_neutral_aug y cross_corpus_induced_4class_test corpus IEMOCAP MSP-IMPROV IEMOCAP_evc "aug_data.datasets.IEMOCAP.subset=IEMOCAP_evc aug_data.datasets.MSP-IMPROV.subset=IEMOCAP_evc" 1324
exp_add_aug y cross_corpus_induced_4class_test corpus IEMOCAP MSP-IMPROV IEMOCAP_evc "aug_data.datasets.IEMOCAP.subset=IEMOCAP_evc aug_data.datasets.MSP-IMPROV.subset=IEMOCAP_evc"

exp_fixed_aug y cross_corpus_induced_4class_test corpus MSP-IMPROV IEMOCAP MSP-IMPROV_evc "aug_data.datasets.IEMOCAP.subset=MSP-IMPROV_evc aug_data.datasets.MSP-IMPROV.subset=MSP-IMPROV_evc"
exp_fixed_neutral_aug y cross_corpus_induced_4class_test corpus MSP-IMPROV IEMOCAP MSP-IMPROV_evc "aug_data.datasets.IEMOCAP.subset=MSP-IMPROV_evc aug_data.datasets.MSP-IMPROV.subset=MSP-IMPROV_evc" 2911
exp_add_aug y cross_corpus_induced_4class_test corpus MSP-IMPROV IEMOCAP MSP-IMPROV_evc "aug_data.datasets.IEMOCAP.subset=MSP-IMPROV_evc aug_data.datasets.MSP-IMPROV.subset=MSP-IMPROV_evc"

exp_real y cross_corpus_induced_4class_test corpus IEMOCAP MSP-IMPROV
exp_real y cross_corpus_induced_4class_test corpus MSP-IMPROV IEMOCAP
exp_real_neutral y cross_corpus_induced_4class_test corpus IEMOCAP MSP-IMPROV 1324
exp_real_neutral y cross_corpus_induced_4class_test corpus MSP-IMPROV IEMOCAP 2911


## IEMOCAP within-corpus session split
exp_manual IEMOCAP_test IEMOCAP IEMOCAP IEMOCAP_evc "aug_data.datasets.IEMOCAP.subset=IEMOCAP_evc"

## MSP-IMPROV within-corpus session split
exp_manual MSP-IMPROV_test MSP-IMPROV MSP-IMPROV MSP-IMPROV_evc "aug_data.datasets.MSP-IMPROV.subset=MSP-IMPROV_evc"

## CREMA-D within-corpus speaker split
exp_manual CREMA-D_test CREMA-D CREMA-D CREMA-D_evc "aug_data.datasets.CREMA-D.subset=CREMA-D_evc"

## EmoV-DB within-corpus speaker split
exp_manual EmoV-DB_test EmoV-DB EmoV-DB EmoV-DB_evc "aug_data.datasets.EmoV-DB.features=datasets/EmoV-DB_aug/EmoV-DB_evc/${features}.nc"
