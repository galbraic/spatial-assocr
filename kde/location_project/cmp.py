import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from plotly.offline import iplot
import plotly.graph_objs as go

from .kde_2d import kdnearest, learn_nearest_neighbors_bandwidth, sample_from_kde


def get_individual_component(df, users, criteria=None, prnt=True):
    # user's location set is all unique parcels visited regardless of month
    if criteria is None:
        criteria = df["uid"].isin(users)
    loc_set = df.loc[criteria]["location_id"].unique()

    # sample space is any other users' data in either month
    samp_sp = df.loc[~df.uid.isin(users)]

    # loop over users in sample space counting number of overlapped parcels
    matches = []
    for u in samp_sp.uid.unique():
        tmp = samp_sp.loc[samp_sp.uid == u]
        shared_locs = np.intersect1d(loc_set, tmp["location_id"].unique())
        if len(shared_locs) > 0:
            matches.append({"uid": u, "n_matches": len(shared_locs), "n_events": len(tmp)})
    matches = pd.DataFrame(matches)

    # compute the weight for each matching user's points
    tot_matches = sum(matches["n_matches"])
    matches["w"] = matches["n_matches"] / (tot_matches * matches["n_events"])
    matches.drop(["n_events", "n_matches"], axis=1, inplace=True)

    # limit the sample space
    samp_sp = pd.merge(samp_sp, matches, on="uid")
    if prnt:
        print("USERS", users)
        print("Number of unique locations (across A & B):", len(loc_set))
        print(
            "Number of locations in common (in both A & B):",
            sum(
                df.loc[criteria]
                .drop_duplicates(subset=["location_id", "m"])
                .groupby("location_id")["uid"]
                .count()
                > 1
            ),
        )
        print("Number of matched users:", samp_sp.uid.nunique())
        print("Number of sample points:", len(samp_sp))
        print("Sum of weights:", round(sum(samp_sp.w), 2))
        print("")

    # learn the bw & format for sampling
    pts = samp_sp.loc[:, ["lon", "lat"]].values
    samp_sp["bw"] = learn_nearest_neighbors_bandwidth(pts, k=5, min_bw=0.05)
    samp_sp = samp_sp[["uid", "lon", "lat", "bw", "w"]]  # reorder

    return samp_sp, np.array(samp_sp)


def sample_from_mixture_kde(pop, indiv, n, users, alpha=0.2):
    # setup population kde with equal weights
    tmp_pop = pop.copy()
    tmp_pop = pop[np.isin(pop[:, 0], users), :]  # remove user from the population
    n_pts = len(tmp_pop)
    pop_w = np.ones(n_pts) / n_pts
    pop_kde = np.hstack([tmp_pop, np.atleast_2d(pop_w).T])

    n_pop = np.random.binomial(n, alpha)

    sample_pop = sample_from_kde(pop_kde, n=n_pop)
    sample_indiv = sample_from_kde(indiv, n=n - n_pop)

    return np.vstack([sample_pop, sample_indiv])


def calc_cmp(mpp, pop_kde_data, userA, userB=None, n_sim=10, k=1, prnt=False):

    if not userB:  # doing it for the same source
        userB = userA

    # get the sample space
    criteria_A = (mpp["m"] == "a") & (mpp["uid"] == userA)
    criteria_B = (mpp["m"] == "b") & (mpp["uid"] == userB)
    indiv_data, indiv_kde_data = get_individual_component(
        df=mpp, users={userA, userB}, criteria=criteria_A | criteria_B, prnt=prnt
    )
    indiv_data.head()

    # SET MPP OF INTEREST & ITS SUBPROCESSES
    a_star = mpp.loc[criteria_A]  # fix events in A^*
    b_star = mpp.loc[criteria_B]
    a_star_unique = a_star.drop_duplicates(subset="location_id")
    b_star_unique = b_star.drop_duplicates(subset="location_id")

    # PERFORM THE SIMULATION
    sim = {}
    scores = {}
    for ell in range(n_sim):
        sim[ell] = {}
        sim[ell]["locations"] = sample_from_mixture_kde(
            pop=pop_kde_data,
            indiv=indiv_kde_data,
            n=len(b_star_unique),
            # n=n_b_star,
            users=[userA, userB],
        )
        s = pd.DataFrame(sim[ell]["locations"], columns=["lon", "lat"])
        sim[ell]["dists"] = kdnearest(a=a_star_unique, b=s, k=k)
        # sim[ell]['dists'] = kdnearest(a=a_star, b=s, k=k)

        scores[ell] = {}
        scores[ell]["mean_dist"] = np.mean(sim[ell]["dists"])
        scores[ell]["med_dist"] = np.median(sim[ell]["dists"])
        # scores[ell]['loc_sim'] = calc_location_similarity(
        #     pd.concat(
        #         [
        #             a_star,
        #             pd.concat(
        #                 [
        #                     b_star_unique[['old_uid','uid','m','location_id']],
        #                     s
        #                 ],
        #                 axis=1
        #             )
        #         ],
        #         ignore_index=True
        #     )
        # )

    scores = pd.DataFrame.from_dict(scores, orient="index")
    obs_dist = kdnearest(a=a_star_unique, b=b_star_unique, k=k)
    # obs_dist = kdnearest(a=a_star, b=b_star, k=k)
    # obs_loc = calc_location_similarity(
    #     pd.concat([a_star, b_star], ignore_index=True)
    # )

    cmp_mean = sum(scores.mean_dist < np.mean(obs_dist)) / n_sim
    cmp_median = sum(scores.med_dist < np.mean(obs_dist)) / n_sim
    # cmp_loc = sum(scores.loc_sim > obs_loc) / n_sim

    return cmp_mean, cmp_median, np.mean(obs_dist), np.median(obs_dist)


def get_user(mpp, uid, mark):
    out = mpp.copy()
    out = out.drop_duplicates(subset=["location_id", "m"])
    return out.loc[(out.uid == uid) & (out.m == mark)].reset_index(drop=True)


def make_user_scatter_plot(mpp, uid, mark):
    user = get_user(mpp, uid, mark)
    name = "m={}, n={}".format(mark.upper(), len(user))
    return go.Scatter(x=user.lon, y=user.lat, mode="markers", name=name, visible="legendonly")


def plot_scatter(df, mpp, uid):
    # calculate counts for heatmap
    data = df.copy()
    data["lat_r"] = data.lat.round(3)
    data["lon_r"] = data.lon.round(3)
    freqs = data.groupby(["lat_r", "lon_r"]).count().reset_index()[["lat_r", "lon_r", "event_id"]]
    freqs.columns = ["lat_r", "lon_r", "freq"]
    p_freqs = freqs.pivot_table(columns="lon_r", index="lat_r", values="freq", fill_value=0)

    # make the heatmap
    heat = go.Heatmap(
        z=gaussian_filter(p_freqs.values, sigma=0.5).tolist(),
        x=p_freqs.columns,
        y=p_freqs.index,
        colorscale=[
            [0, "rgb(0, 0, 0)"],  # 0
            [1.0 / 10000, "rgb(0, 0, 0)"],
            [1.0 / 10000, "rgb(100, 100, 100)"],  # 10
            [1.0 / 1000, "rgb(130, 130, 130)"],  # 100
            [1.0 / 100, "rgb(170, 170, 170)"],  # 1000
            [1.0 / 10, "rgb(220, 220, 220)"],  # 10000
            [1.0, "rgb(255, 255, 255)"],  # 100000
        ],
        colorbar={"tick0": 0, "tickmode": "array", "tickvals": [0, 1000, 10000, 100000]},
    )

    traces = [heat]
    traces += [make_user_scatter_plot(mpp, uid, mark) for mark in ["a", "b"]]

    layout = go.Layout(title="User {}".format(uid), legend=dict(x=-0.35, y=1))
    fig = go.Figure(data=traces, layout=layout)

    iplot(fig, show_link=False)
