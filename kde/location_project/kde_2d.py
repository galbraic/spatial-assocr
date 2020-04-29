"""
Author: Moshe Lichman
"""
# from __future__ import division
import numpy as np
import pandas as pd
from plotly.offline import iplot
import plotly.graph_objs as go
from scipy.special import logsumexp
from scipy.spatial import KDTree
from scipy.stats import norm

MIN_PARENT = 5
TOL = 0.0001

# # Might need adjusting for different areas - this one is for SoCal
# KM_TO_LON = 0.010615
# KM_TO_LAT = 0.008989

# # this one is for Lusanne, Switzerland
# KM_TO_LON = 0.01306
# KM_TO_LAT = 0.009016

# this one is for NY
KM_TO_LON = 0.011856
KM_TO_LAT = 0.009043


class KDE(object):
    def __init__(self, data):
        """
         INPUT:
        -------
            1. data: The observed/train events.
                     The data is in the format of
                     np.array([[lon, lat, bandwidth, weight], ... ])
                     where each line is a different point.
        """
        self.data = data

    def log_pdf(self, query_point):
        """
         INPUT:
        -------
            :param query_point: single point np.array([[lon, lat]])

         OUTPUT:
        --------
            1. log_pdf: <float> the log pdf value
        """
        x_dist = query_point[0] - self.data[:, 0]
        y_dist = query_point[1] - self.data[:, 1]
        log_pdf_x = norm.logpdf(x_dist, loc=0, scale=self.data[:, 2] * KM_TO_LON)
        log_pdf_y = norm.logpdf(y_dist, loc=0, scale=self.data[:, 2] * KM_TO_LAT)
        return logsumexp(log_pdf_x + log_pdf_y) - np.log(self.data.shape[0])

    def log_lik(self, query_points):
        """
         INPUT:
        -------
            :param query_points: array of points np.array([[lon, lat]], ...)

         OUTPUT:
        --------
            1. log_pdf: <float> the log likelihood value
        """
        return sum(np.apply_along_axis(self.log_pdf, 1, query_points))


class MixtureKDE(object):
    def __init__(self, indiv, pop, alpha=0.80):
        """
         INPUT:
        -------
            1. indiv: The observed/train events for the local component
                     The data is in the format of
                     np.array([[lon, lat, bandwidth, weight], ... ])
                     where each line is a different point.
            2. population: The observed/train events for the population component
                     The data is in the format of
                     np.array([[lon, lat, bandwidth, weight], ... ])
                     where each line is a different point.
            3. alpha: (default=0.80) The mixing weight for the individual.
                     The population will get 1-alpha
        """
        self.indiv = KDE(indiv)
        self.pop = KDE(pop)
        self.alpha = alpha

    def log_pdf(self, query_point):
        indiv_log_pdf = np.log(self.alpha) + self.indiv.log_pdf(query_point)
        pop_log_pdf = np.log(1 - self.alpha) + self.pop.log_pdf(query_point)
        return np.logaddexp(indiv_log_pdf, pop_log_pdf)

    def log_lik(self, query_points):
        return sum(np.apply_along_axis(self.log_pdf, 1, query_points))


def learn_nearest_neighbors_bandwidth(
    sample_points, k=5, lon_to_km=KM_TO_LON, lat_to_km=KM_TO_LAT, min_bw=0.05
):
    """
     INPUT:
    -------
        :param sample_points:
        :param k:
        :param lon_to_km:
        :param lat_to_km:
        :param min_bw:

     OUTPUT:
    --------
        :return: np.array of bandwidths
    """
    if sample_points.shape[0] == 1:
        return np.array(1.0)

    k = np.min([k, sample_points.shape[0] - 1])

    bandwidths = []

    anchor_point = np.amin(sample_points, 0)
    dists = sample_points - anchor_point

    dists[:, 0] /= lon_to_km
    dists[:, 1] /= lat_to_km

    dists += np.random.random(dists.shape) * 0.0000001

    # Building the k-d tree
    tree = KDTree(dists, leafsize=500)

    for i in range(dists.shape[0]):
        (neighbors_dists, neighbors_indexes) = tree.query(dists[i, :], k + 1)

        if neighbors_dists[-1] <= min_bw:
            bandwidths.append(min_bw)
        else:
            bandwidths.append(neighbors_dists[-1])

    # print("Done training bandwidths")
    return np.array(bandwidths)


def kdnearest(a, b, leafsize=10, k=1):
    """
    Measure the Haversine distance from each point in A
    to its nearest neighbor in B using a KDTree data struct
    built from the locations in B.

    Params:
        a: pandas.DataFrame containing <'lat', 'lon'>
        b: pandas.DataFrame containing <'lat', 'lon'>
        leafsize: int specifing the number of data points to
            place in each leaf node of the KDTree
        k: int number of nearest nei
    Returns:
        pandas.DataFrame of distances in km
    """
    a_copy = a.copy()
    b_copy = b.copy()
    R = 6371  # radius of earth in km

    def dist_to_arclength(chord_length):
        """
        https://en.wikipedia.org/wiki/Great-circle_distance
        Convert Euclidean chord length to great circle arc length
        """
        central_angle = 2 * np.arcsin(chord_length / (2.0 * R))
        arclength = R * central_angle
        return arclength

    def to_cartesian(data):
        """convert to Cartesian coordinates"""
        phi = np.deg2rad(data["lat"])
        theta = np.deg2rad(data["lon"])
        data["x"] = R * np.cos(phi) * np.cos(theta)
        data["y"] = R * np.cos(phi) * np.sin(theta)
        data["z"] = R * np.sin(phi)
        return data[["x", "y", "z"]]

    # build the tree and query it
    btree = KDTree(to_cartesian(b_copy), leafsize=leafsize)
    dist, idx = btree.query(to_cartesian(a_copy), k=k)
    if k > 1:
        return dist_to_arclength(dist)[:, k - 1]
    return dist_to_arclength(dist)


def sample_from_kde(data, n=1):
    """
    Inputs:
        data: np.array([[user_id, lon, lat, bandwidth, weight], ... ])
        n: number of samples
    Output:
        np.array of sampled points: np.array([[lon, lat], ...])
    """
    sampled_point_index = np.random.choice(data.shape[0], size=n, p=data[:, -1])
    sampled_points = data[sampled_point_index, :]

    sample_lon = np.random.normal(sampled_points[:, 1], sampled_points[:, 3] * KM_TO_LON)
    sample_lat = np.random.normal(sampled_points[:, 2], sampled_points[:, 3] * KM_TO_LAT)

    return np.vstack([sample_lon, sample_lat]).T


def make_user_scatter_plot(user_data, mark):
    dat = user_data.loc[user_data.m == mark]
    name = "m={}, n={}".format(mark, len(dat))
    return go.Scatter(
        x=dat.lon,
        y=dat.lat,
        mode="markers",
        marker=dict(color="rgb(0, 0, 0)" if mark == "a" else "rgb(0,128,0)"),
        name=name,
        visible="legendonly",
    )


def plot_kde(kde, user_data=None, uid=None):
    """
    Compute log pdf values over grid

    data: np.array([[lon, lat, bw], ...])
    """
    # evaluate density over OC
    delta = 0.01
    x = np.arange(-118.2, -117.5, delta)  # longitude
    y = np.arange(33.4, 34, delta)  # latitude
    X, Y = np.meshgrid(x, y)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    z = np.apply_along_axis(kde.log_pdf, 1, pts)
    out = pd.DataFrame({"lon": pts[:, 0], "lat": pts[:, 1], "lpdf": z})

    # plot the heatmap
    heat = go.Heatmap(
        z=out.lpdf,
        x=out.lon,
        y=out.lat,
        colorscale=[
            [1.0, "rgb(165,0,38)"],
            [0.8888888888888888, "rgb(215,48,39)"],
            [0.7777777777777778, "rgb(244,109,67)"],
            [0.6666666666666666, "rgb(253,174,97)"],
            [0.5555555555555556, "rgb(254,224,144)"],
            [0.4444444444444444, "rgb(224,243,248)"],
            [0.3333333333333333, "rgb(171,217,233)"],
            [0.2222222222222222, "rgb(116,173,209)"],
            [0.1111111111111111, "rgb(69,117,180)"],
            [0.0, "rgb(49,54,149)"],
        ],
    )

    traces = [heat]
    layout = go.Layout()

    if user_data is not None:
        traces += [make_user_scatter_plot(user_data, mark) for mark in ("a", "b")]

        layout = go.Layout(title="User {}".format(uid), legend=dict(x=-0.35, y=1))

    fig = go.Figure(data=traces, layout=layout)
    iplot(fig, show_link=False)


def create_individual_component_data(df):
    points = df[["lon", "lat"]].values
    bw = learn_nearest_neighbors_bandwidth(points, k=5, min_bw=0.01)
    indiv_kde = np.hstack([points, np.atleast_2d(bw).T])
    indiv_kde_data = np.append(indiv_kde, np.ones((len(indiv_kde), 1)), 1)

    return indiv_kde_data
