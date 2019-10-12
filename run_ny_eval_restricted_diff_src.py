#!/usr/bin/env python

# flake8: noqa: E999
import numpy as np
import pandas as pd
import os
from scipy import stats


def get_diff_src_set(scores, mpp, uA, uB):
    uA_locs = mpp.loc[(mpp["uid"] == uA) & (mpp["m"] == "a"), "location_id"].unique()
    samp_sp = mpp.loc[~mpp.uid.isin({uA, uB}) & mpp.location_id.isin(uA_locs), "uid"].unique()
    return scores.loc[
        (
            (scores.a != scores.b)
            & ~(scores["a"].isin({uA, uB}))
            & ~(scores["b"].isin({uA, uB}))
            & (scores["a"].isin(samp_sp) | scores["b"].isin(samp_sp))
        ),
    ]


def calc_slr(df, uA, uB, score, diff=None):
    val = df.loc[(df.a == uA) & (df.b == uB), score].to_numpy()
    same = df.loc[(df.a == df.b) & ~(df.a.isin({uA, uB})), score].to_numpy()
    if diff is None:
        diff = df.loc[
            (df.a != df.b) & ~(df.a.isin({uA, uB})) & ~(df.b.isin({uA, uB})), score
        ].to_numpy()
    else:
        diff = diff[score].to_numpy()

    kde_same = stats.gaussian_kde(same)
    kde_diff = stats.gaussian_kde(diff)

    return (kde_same.evaluate(val) / kde_diff.evaluate(val))[0]


def calc_cmp(df, uA, uB, score, diff=None):
    val = df.loc[(df.a == uA) & (df.b == uB), score].to_numpy()
    if diff is None:
        diff = df.loc[
            (df.a != df.b) & ~(df.a.isin({uA, uB})) & ~(df.b.isin({uA, uB})), score
        ].to_numpy()
    else:
        diff = diff[score].to_numpy()

    return sum(diff < val) / diff.shape[0]


def main():
    # read in necessary data
    directory = os.path.join(os.getcwd(), "data", "ny")
    mpp = pd.read_csv(os.path.join(directory, "mpp_visits_month0a_month1b_n20.csv"))
    score_vals = pd.read_csv(os.path.join(directory, "population_score_func_vals.csv"))

    # randomly sample the diff-src pairs for computational efficiency
    d = score_vals.loc[score_vals.a != score_vals.b, ["a", "b"]].sample(10000, random_state=1234)
    sample = pd.concat([score_vals.loc[score_vals.a == score_vals.b, ["a", "b"]], d]).reset_index(
        drop=True
    )

    # these are the scores we want o compute slr & cmp for
    scores = [
        "ied_med",
        "ied_mn",
        "ied_mn_wt_event",
        "ied_mn_wt_user",
        "emd",
        "emd_wt_event",
        "emd_wt_user",
    ]

    # perform the computations
    slr = []
    cmp = []
    for index, row in sample.iterrows():
        if index % round(len(sample) / 10) == 0:
            print(round(index / len(sample), 1) * 100, "%")
        try:
            tmp_slr = {"a": row["a"], "b": row["b"]}
            tmp_cmp = {"a": row["a"], "b": row["b"]}
            diff = get_diff_src_set(score_vals, mpp, row["a"], row["b"])
            for s in scores:
                tmp_slr[s] = calc_slr(score_vals, row["a"], row["b"], s, diff)
                tmp_cmp[s] = calc_cmp(score_vals, row["a"], row["b"], s, diff)
            slr.append(tmp_slr)
            cmp.append(tmp_cmp)
        except:
            continue

    # convert to df
    slr = pd.DataFrame.from_dict(slr)
    cmp = pd.DataFrame.from_dict(cmp)

    # write to file
    slr.to_csv(os.path.join(directory, "slr_restricted_diff_src.csv"), index=False)
    cmp.to_csv(os.path.join(directory, "cmp_population_restricted_diff_src.csv"), index=False)

    print("Successfully finished!")


if __name__ == "__main__":
    main()
