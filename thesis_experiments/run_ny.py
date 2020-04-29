#!/usr/bin/env python

# flake8: noqa: E999
import numpy as np
import pandas as pd
import os
from scipy import stats

from kde.location_project.kde_2d import KDE, MixtureKDE, create_individual_component_data


def calc_slr(df, uA, uB, score):
    val = df.loc[(df.a == uA) & (df.b == uB), score].to_numpy()
    same = df.loc[(df.a == df.b) & ~(df.a.isin({uA, uB})), score].to_numpy()
    diff = df.loc[
        (df.a != df.b) & ~(df.a.isin({uA, uB})) & ~(df.b.isin({uA, uB})), score
    ].to_numpy()

    kde_same = stats.gaussian_kde(same)
    kde_diff = stats.gaussian_kde(diff)

    return (kde_same.evaluate(val) / kde_diff.evaluate(val))[0]


def calc_cmp(df, uA, uB, score):
    val = df.loc[(df.a == uA) & (df.b == uB), score].to_numpy()
    diff = df.loc[
        (df.a != df.b) & ~(df.a.isin({uA, uB})) & ~(df.b.isin({uA, uB})), score
    ].to_numpy()

    return sum(diff < val) / diff.shape[0]


def calculate_lr(mpp, pop_kde_data, uA, uB, alpha=0.80):
    # get index of points in patterns A and B, respectively
    ind_A = (mpp.uid == uA) & (mpp.m == "a")
    ind_B = (mpp.uid == uB) & (mpp.m == "b")

    # create the mixture KDE
    i_kde_data = create_individual_component_data(mpp.loc[ind_A,])
    i_kde = MixtureKDE(i_kde_data, pop_kde_data, alpha=alpha)

    # evaluate the points in question (pattern B)
    eval_points = mpp.loc[ind_B, ["lon", "lat"]].values
    pop_kde = KDE(pop_kde_data)

    return i_kde.log_lik(eval_points) - pop_kde.log_lik(eval_points)


def compute_weight_step(n_a):
    if n_a <= 5:
        alpha = 0.05
    elif n_a > 5 and n_a <= 10:
        alpha = 0.15
    elif n_a > 10 and n_a <= 20:
        alpha = 0.40
    elif n_a > 20 and n_a <= 50:
        alpha = 0.55
    elif n_a > 50 and n_a <= 100:
        alpha = 0.70
    elif n_a > 100:
        alpha = 0.85

    return alpha


def compute_weight_func(n_a):
    return (1 + np.exp(-0.02 * n_a)) ** -1 - 0.35 * n_a ** (-1 / 2) - 0.1


def main():
    np.random.seed(1234)

    # read in necessary data
    directory = os.path.join("..", "data", "ny")
    mpp = pd.read_csv(os.path.join(directory, "mpp_visits_month0a_month1b_n1.csv"))
    pop_kde_data = np.load(os.path.join(directory, "population_visits_kde_data.npy"))
    score_vals = pd.read_csv(os.path.join(directory, "score_func_vals.csv"))
    score_vals.sort_values(["a", "b"], inplace=True)
    score_vals = score_vals.reset_index()

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
    lr = []
    for index, row in score_vals.iterrows():
        if index % round(len(score_vals) / 10) == 0:
            print(round(index / len(score_vals), 1) * 100, "%")
        tmp_slr = {"a": row["a"], "b": row["b"]}
        tmp_cmp = {"a": row["a"], "b": row["b"]}
        tmp_lr = {"a": row["a"], "b": row["b"]}

        # score-based methods
        for s in scores:
            tmp_slr[s] = calc_slr(score_vals, row["a"], row["b"], s)
            tmp_cmp[s] = calc_cmp(score_vals, row["a"], row["b"], s)

        # LR with different weighting schemes
        tmp_lr["lr_alpha_80"] = calculate_lr(mpp, pop_kde_data, uA=row["a"], uB=row["b"], alpha=0.8)

        n_a = len(mpp.loc[(mpp.uid == row["a"]) & (mpp.m == "a"),])
        tmp_lr["alpha_step"] = compute_weight_step(n_a)
        tmp_lr["lr_alpha_step"] = calculate_lr(
            mpp, pop_kde_data, uA=row["a"], uB=row["b"], alpha=tmp_lr["alpha_step"]
        )

        tmp_lr["alpha_func"] = compute_weight_func(n_a)
        tmp_lr["lr_alpha_func"] = calculate_lr(
            mpp, pop_kde_data, uA=row["a"], uB=row["b"], alpha=tmp_lr["alpha_step"]
        )

        slr.append(tmp_slr)
        cmp.append(tmp_cmp)
        lr.append(tmp_lr)

    # convert to df
    slr = pd.DataFrame.from_dict(slr)
    cmp = pd.DataFrame.from_dict(cmp)
    lr = pd.DataFrame.from_dict(lr)

    # write to file
    slr.to_csv(os.path.join(directory, "slr.csv"), index=False)
    cmp.to_csv(os.path.join(directory, "cmp.csv"), index=False)
    lr.to_csv(os.path.join(directory, "lr.csv"), index=False)

    print("Successfully finished!")


if __name__ == "__main__":
    main()
