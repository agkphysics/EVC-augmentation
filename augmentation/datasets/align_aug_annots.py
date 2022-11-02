from pathlib import Path
from typing import Tuple

import click

from ertk.dataset import (
    CorpusInfo,
    Dataset,
    SubsetInfo,
    write_annotations,
    write_filelist,
)


@click.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path))
@click.option("--original", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--annot", "annots", multiple=True)
def main(input: Path, original: Path, annots: Tuple[str]):
    names = [x.stem for x in sorted(set(input.glob("**/*.wav")))]
    orig_names = [x.rsplit("_", maxsplit=1)[0] for x in names]
    write_filelist([f"{x}.wav" for x in names], input / "files_all.txt")

    orig_data = Dataset(original, subset="all")
    orig_annots = orig_data.annotations.loc[orig_names]

    corpus_info = CorpusInfo(name=orig_data.corpus)
    corpus_info.partitions = ["label", "original"]
    corpus_info.subsets["all"] = SubsetInfo("files_all.txt")

    for d in input.glob("*/"):
        path = d / "aug.txt"
        if path.exists():
            corpus_info.subsets[d.stem] = SubsetInfo(str(path.relative_to(input)))

    labels = {x: x.split("_")[-1] for x in names}
    write_annotations(labels, "label", input / "label.csv")
    orig = dict(zip(names, orig_names))
    write_annotations(orig, "original", input / "original.csv")
    for annot in annots:
        corpus_info.partitions.append(annot)
        orig_annot = dict(zip(names, orig_annots[annot]))
        write_annotations(orig_annot, annot, input / f"{annot}.csv")
    CorpusInfo.to_file(corpus_info, input / "corpus.yaml")


if __name__ == "__main__":
    main()
