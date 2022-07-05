#!/usr/bin/env python

import os
import sys
import glob
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from metrics import (
    roc_auc,
    pr_auc,
    precision_at_L,
    recall_at_L,
    bedroc_score,
    max_f1,
    max_mcc,
    max_balanced_accuracy,
)

__version__ = "20220511"


def define_header(filepath):
    filename = os.path.basename(filepath)
    _, _, parameters, _ = filename.split(".")
    parameters = parameters.split("_")

    header = [
        "Dataset",
        "Method",
        "CV",
        "Fold ID",
        "Fingerprint" if len(parameters) == 1 else None,
        "Alpha" if len(parameters) == 2 else None,
        "ROC-AUC",
        "PR-AUC",
        "P(20)",
        "R(20)",
        "BEDROC(20)",
        "Max Balanced Accuracy",
        "Max MCC",
        "Max F1 score",
    ]

    return ",".join(filter(lambda v: v is not None, header))


def evaluate(filepath):
    print(filepath)
    # Read predictions output file and extract metadata
    df = pd.read_csv(filepath, sep=",", names=["fold", "ligand", "target", "score", "tp"])
    df = df.drop_duplicates(ignore_index=True)
    df = df[df.score != -99]

    # Get predictions metadata from file name
    filename = os.path.basename(filepath)
    dataset, scenario, parameters, _ = filename.split(".")
    method, cv = scenario.split("_")

    # Retrieve parameters depending in method
    parameters = parameters.split("_")

    # Calculate metrics per fold
    metrics = pd.DataFrame()
    for fidx in df["fold"].unique():
        f_df = df[df["fold"] == int(fidx)]
        f_eval = {}

        # Metadata
        f_eval["Dataset"] = dataset
        f_eval["Method"] = method
        f_eval["CV"] = cv
        f_eval["Fold ID"] = fidx
        if len(parameters) == 1:
            fp = parameters
            f_eval["Fingerprint"] = fp
        elif len(parameters) == 2:
            fp, alpha = parameters
            f_eval["Fingerprint"] = fp
            f_eval["Alpha"] = float(alpha[:1] + "." + alpha[1:])

        # Metrics
        ## Classic metrics
        f_eval["ROC-AUC"] = roc_auc(f_df.tp, f_df.score)
        f_eval["PR-AUC"] = pr_auc(f_df.tp, f_df.score)
        ## Early-reognition
        f_eval["P(20)"] = precision_at_L(f_df, 20)
        f_eval["R(20)"] = recall_at_L(f_df, 20)
        f_eval["BEDROC(20)"] = bedroc_score(f_df.tp.values, f_df.score.values, alpha=20)
        ## Binary classification "best-case-scenario"
        f_eval["Max Balanced Accuracy"] = max_balanced_accuracy(
            f_df.tp.values, f_df.score.values
        )[0]
        f_eval["Max MCC"] = max_mcc(f_df.tp.values, f_df.score.values)[0]
        f_eval["Max F1 score"] = max_f1(f_df.tp.values, f_df.score.values)[0]

        # print(f_eval)
        metrics = metrics.append(pd.DataFrame(f_eval, index=[0]))

    return metrics


def main():
    FILES2EVAL = list(glob.glob(str(sys.argv[1])))

    # Create file and add header
    file_pattern = FILES2EVAL[0]
    eval_filepath = sys.argv[2] + f".eval_v{__version__}.csv"
    with open(eval_filepath, "w") as f:
        f.write(define_header(file_pattern) + "\n")

    # Parallel evaluation of prediction
    pbar = tqdm(desc="Evaluating predictive performance", total=len(FILES2EVAL))
    pool = Pool()
    for file_eval in pool.imap_unordered(evaluate, FILES2EVAL):
        file_eval.to_csv(
            eval_filepath,
            mode="a+",
            header=False,
            index=False,
        )
        pbar.update(1)
    pool.close()
    pool.join()


if __name__ == "__main__":
    # execute only if run as a script
    main()
