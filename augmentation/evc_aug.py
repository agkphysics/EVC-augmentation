import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ertk.classification import get_balanced_sample_weights, standard_class_scoring
from ertk.config import get_arg_mapping
from ertk.dataset import DataLoadConfig, load_datasets_config
from ertk.sklearn.models import get_sk_model
from ertk.train import ExperimentConfig, get_cv_splitter, get_scores


@dataclass
class AugExperimentConfig(ExperimentConfig):
    aug_data: Optional[DataLoadConfig] = None


@click.command()
@click.option("--config", "config_path", type=Path)
@click.option("--p_real", type=float)
@click.option("--p_fake", type=float, help="Ratio of fake:real data.")
@click.option("--max_train", type=float, help="Max train instances")
@click.option("--results", type=Path)
@click.option("--reps", type=int, default=1)
@click.option("--eval")
@click.option("--fake_method", default="full")
@click.option("--debug", is_flag=True)
@click.argument("restargs", nargs=-1)
def main(
    config_path: Path,
    p_real: float,
    p_fake: float,
    max_train: float,
    results: Path,
    reps: int,
    eval: str,
    fake_method: str,
    debug: bool,
    restargs: Tuple[str],
):
    if debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)
    exp_config = AugExperimentConfig.from_file(config_path, list(restargs))

    print("Loading real data")
    data = load_datasets_config(exp_config.data)
    print(data)

    print("Loading augmented data")
    aug_data = load_datasets_config(exp_config.aug_data)
    print(aug_data)

    eval_config = exp_config.eval
    if eval:
        eval_config = exp_config.evals[eval]

    train_idx = data.get_idx_for_split(eval_config.train)
    test_idx = data.get_idx_for_split(eval_config.test)
    train_names = data.names[train_idx]
    train_x = data.x[train_idx]
    train_y = data.y[train_idx]
    test_x = data.x[test_idx]
    test_y = data.y[test_idx]

    if p_fake < 0:
        train_size = min(len(aug_data), len(train_idx))
        if max_train:
            if 0 < max_train <= 1:
                train_size = int(max_train * train_size)
            else:
                train_size = int(max_train)
        n_real = int(p_real * train_size)
        # n_real = min(n_real, min_train)
    else:
        train_size = len(train_idx)
    n_real = int(p_real * train_size)

    print(f"Selecting from {len(train_idx)} real train instances: {eval_config.train}")
    print(f"Using {len(test_idx)} real test instances: {eval_config.test}")

    transform = StandardScaler()
    model_type = exp_config.model.type.split("/")[1]
    clf = get_sk_model(model_type, **get_arg_mapping(exp_config.model.args_path))
    clf = Pipeline([("transform", transform), ("clf", clf)])
    if exp_config.model.param_grid_path:
        param_grid = get_arg_mapping(exp_config.model.param_grid_path)
        clf = GridSearchCV(
            clf,
            {f"clf__{k}": v for k, v in param_grid.items()},
            scoring="balanced_accuracy",
            cv=get_cv_splitter(False, 2, shuffle=True, random_state=54321),
            n_jobs=exp_config.training.n_jobs,
        )
    print(clf)
    scoring = standard_class_scoring(data.classes)

    rng = np.random.default_rng(54321)

    scores_df = pd.DataFrame(
        index=pd.RangeIndex(reps, name="rep"), columns=list(scoring.keys())
    )
    cms = []
    for rep in range(reps):
        # Select sample of real data
        perm = rng.permutation(len(train_x))[:n_real]

        if fake_method == "full":
            aug_idx = np.arange(len(aug_data.x))
        elif fake_method == "train" and n_real > 0:
            orig_names = aug_data.annotations.loc[aug_data.names, "original"]
            aug_idx = np.flatnonzero(orig_names.isin(train_names[perm]))
        else:
            raise ValueError("fake_method")

        # Select sample of augmented data
        if p_fake > 0:
            n_fake = int(p_fake * len(aug_idx))
        elif p_fake < 0:
            # Make up rest of train data with fake data
            n_fake = train_size - n_real
            if n_fake > len(aug_data):
                n_fake = len(aug_data)
                if rep == 0:
                    warnings.warn("Not enough augmented data to make up train data.")
        else:
            n_fake = 0

        if rep == 0:
            print(f"Using {n_real} real and {n_fake} fake train instances.")

        real_x = train_x[perm]
        real_y = train_y[perm]
        aug_idx = rng.permutation(aug_idx)[:n_fake]
        aug_x = aug_data.x[aug_idx]
        aug_y = aug_data.y[aug_idx]
        train_x_aug = np.concatenate([real_x, aug_x])
        train_y_aug = np.concatenate([real_y, aug_y])
        train_sw = get_balanced_sample_weights(train_y_aug)

        # Train and test
        clf.fit(train_x_aug, train_y_aug, clf__sample_weight=train_sw)
        y_pred = clf.predict(test_x)
        cms.append(
            confusion_matrix(
                test_y, y_pred, labels=np.arange(data.n_classes), normalize="true"
            )
        )
        scores_df.loc[rep] = get_scores(scoring, y_pred, test_y)
    cm = np.stack(cms).mean(0)
    print(cm)
    scores_df["p_real"] = p_real
    scores_df["p_fake"] = p_fake
    scores_df["max_train"] = max_train
    print(scores_df.mean())
    if results:
        results.parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(results)
        print(f"Wrote results to {results}")


if __name__ == "__main__":
    main()
