#!/usr/bin/env python

# flake8: noqa: E999
import numpy as np
import pandas as pd
import os
from emd import emd
from kde.location_project.cmp import geodesic_dist


def get_user(mpp, uid, mark):
    out = mpp.copy()
    return out.loc[(out.uid == uid) & (out.m == mark)].reset_index(drop=True)


def population_scores(mpp):
    users = sorted(mpp["uid"].unique())
    rslt = []

    for user_A in users[:5]:
        print(f"USER {user_A}")
        for user_B in users[:5]:
            tmp = {}
            try:
                # get the data for the pair in question
                tmp["a"] = user_A
                tmp["b"] = user_B
                uA = get_user(mpp, user_A, "a")
                uB = get_user(mpp, user_B, "b")

                # compute weights
                uA_wt_e = (uA["weight_event"] / sum(uA["weight_event"])).to_numpy()
                uA_wt_u = (uA["weight_user"] / sum(uA["weight_user"])).to_numpy()
                uB_wt_e = (uB["weight_event"] / sum(uB["weight_event"])).to_numpy()
                uB_wt_u = (uB["weight_user"] / sum(uB["weight_user"])).to_numpy()

                # compute the distance matrix & IED scores
                dist = geodesic_dist(uA[["lat", "lon"]], uB[["lat", "lon"]])
                ied = dist.min(1)
                tmp["ied_med"] = np.median(ied)
                tmp["ied_mn"] = np.average(ied)
                tmp["ied_mn_wt_event"] = np.average(ied, weights=uA_wt_e)
                tmp["ied_mn_wt_user"] = np.average(ied, weights=uA_wt_u)

                # compute variants of EMD
                tmp["emd"] = emd(distance="precomputed", D=dist)
                tmp["emd_wt_event"] = emd(
                    X_weights=uA_wt_e, Y_weights=uB_wt_e, distance="precomputed", D=dist
                )
                tmp["emd_wt_user"] = emd(
                    X_weights=uA_wt_u, Y_weights=uB_wt_u, distance="precomputed", D=dist
                )

                # store it
                rslt.append(tmp)
            except:  # noqa
                print(f"Error! A = {user_A}, B = {user_B}")
                continue

    return rslt


def main():
    # read in necessary data
    directory = os.path.join(os.getcwd(), "data", "oc")
    deduped = pd.read_csv(os.path.join(directory, "visits_deduped.csv"))
    mpp = pd.read_csv(os.path.join(directory, "mpp_visits_month0a_month1b_n20.csv"))

    # location weights based on number of visits
    locs = deduped.groupby("location_id")["event_id"].count()
    wt_e = pd.DataFrame(locs).reset_index().rename(columns={"event_id": "weight"})
    wt_e["weight"] = 1 / wt_e["weight"]

    # location weights based on number of unique users at location
    locs_users = deduped.groupby(["location_id"])["old_uid"].nunique()
    wt_u = pd.DataFrame(locs_users).reset_index().rename(columns={"old_uid": "weight"})
    wt_u["weight"] = 1 / wt_u["weight"]

    # merge weights in to the point pattern data
    mpp = pd.merge(mpp, wt_e, on="location_id")
    mpp = pd.merge(mpp, wt_u, on="location_id", suffixes=("_event", "_user"))

    # perform the experiment
    rslt = population_scores(mpp)

    # write to file
    out = pd.DataFrame.from_dict(rslt)
    out.to_csv(os.path.join(directory, "population_score_func_vals.csv"), index=False)

    print("Successfully finished!")


if __name__ == "__main__":
    main()
