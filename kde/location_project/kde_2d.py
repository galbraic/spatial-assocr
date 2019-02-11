"""
Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
from scipy import misc
from scipy.spatial import KDTree

MIN_PARENT = 5
TOL = 0.0001

# Might need adjusting for different areas - this one is for SoCal
KM_TO_LON = 0.010615
KM_TO_LAT = 0.008989


def _log_kernel_for_ball_data(query_point, ball_data):
    x_dist = query_point[0] - ball_data[:, 1]
    y_dist = query_point[1] - ball_data[:, 2]
    log_pdf_x = (
        -0.5 * np.log(2 * np.pi)
        - np.log(ball_data[:, 3] * KM_TO_LON)
        - 0.5 * (x_dist * x_dist) / (ball_data[:, 3] * KM_TO_LON * ball_data[:, 3] * KM_TO_LON)
    )

    log_pdf_y = (
        -0.5 * np.log(2 * np.pi)
        - np.log(ball_data[:, 3] * KM_TO_LAT)
        - 0.5 * (y_dist * y_dist) / (ball_data[:, 3] * KM_TO_LAT * ball_data[:, 3] * KM_TO_LAT)
    )

    log_vals = np.log(ball_data[:, 4]) + log_pdf_x + log_pdf_y

    return misc.logsumexp(log_vals)


def _log_kernel_for_single_sample_event(
    query_point, sample_point, exp_bandwidth, factor_bandwidth, weight
):
    """
     INPUT:
    -------
        :param query_point:
        :param sample_point:
        :param exp_bandwidth:
        :param factor_bandwidth:
        :param weight:

     OUTPUT:
    --------
        1. log_pdf: <float> The log pdf value
    """

    x_dist = query_point[0] - sample_point[0]
    y_dist = query_point[1] - sample_point[1]
    log_pdf_x = (
        -0.5 * np.log(2 * np.pi)
        - np.log(factor_bandwidth * KM_TO_LON)
        - 0.5 * (x_dist * x_dist) / (exp_bandwidth * KM_TO_LON * exp_bandwidth * KM_TO_LON)
    )

    log_pdf_y = (
        -0.5 * np.log(2 * np.pi)
        - np.log(factor_bandwidth * KM_TO_LAT)
        - 0.5 * (y_dist * y_dist) / (exp_bandwidth * KM_TO_LAT * exp_bandwidth * KM_TO_LAT)
    )

    return np.log(weight) + log_pdf_x + log_pdf_y


def _learn_nearest_neighbors_bandwidth(sample_points, k, lon_to_km, lat_to_km):
    """
    Learning the bandwidth
    :param data:
    :param k:
    :return:
    """
    k = np.min([k, sample_points.shape[0] - 1])

    bandwidths = []

    anchor_point = np.amin(sample_points, 0)
    dists = sample_points - anchor_point
    dists[:, 0] /= lon_to_km
    dists[:, 1] /= lat_to_km

    # Building the k-d tree
    tree = KDTree(dists, leafsize=500)

    for i in range(dists.shape[0]):
        (neighbors_dists, neighbors_indexes) = tree.query(dists[i, :], k + 1)

        if neighbors_dists[-1] <= 0.001:
            bandwidths.append(0.001)  # bandwidth can't be less than 1 meter
        else:
            bandwidths.append(neighbors_dists[-1])

        # bandwidths.append(neighbors_dists[-1])

    # print('Done training bandwidths')
    return np.array(bandwidths)


def _build_adaptive_bandwidth_kde(sample_points, nn=20, lon_to_km=KM_TO_LON, lat_to_km=KM_TO_LAT):
    """
    Creates a fast KDE (using k-d tree) model for the sample points.

     INPUT:
    -------
        1. sample_points: The observed data
        2. nn:            (default=20) Number of neighbors to look at when
                            computing the adaptive bandwidth
        3. lon_to_km:     (default=0.010615) What value of longitude is
                            equivalent to 1 km in this region
        4. lat_to_km      (default=0.008989) What value of latitude is
                            equivalent to 1 km in this region

     OUTPUT:
    --------
        1. kde: KDE object with adaptive bandwidth method
    """
    print("Computing the bw")
    bw = _learn_nearest_neighbors_bandwidth(sample_points[:, 1:3], nn, lon_to_km, lat_to_km)

    print("Done computing bw, creating the kd-tree")
    # Combining the sample points with the learned badnwidths
    data = np.hstack((sample_points, np.array([bw]).T, np.ones([sample_points.shape[0], 1])))

    return KDE(data)


class MixtureKdeIndividualAndPopulation(object):
    def __init__(
        self, sample_points, user_id, alpha=0.85, nn=20, lon_to_km=KM_TO_LON, lat_to_km=KM_TO_LAT
    ):
        """
        Creates a mixture of KDE model. This version only has two components:
        the individual model and the population model.

         INPUT:
        -------
            1. sample_points:   All observation points <np.array [[user_id, lon, lat], ... ]
            2. user_id:         The user to create the mixture model for
            3. alpha:           (default=0.85) The mixing weight for the individual. The
                                    population will get 1-alpha
            4. nn:              (default=20) The number of nearest neighbors to compute the
                                    adaptive bandwidth
            5. lon_to_km:       What value of longitude is equivalent to 1 km in this region
            6. lat_to_km        What value of latitude is equivalent to 1 km in this region
        """
        self._all_data = sample_points
        self._user_id = user_id
        self._alpha = alpha
        user_data = sample_points[np.where(sample_points[:, 0] == user_id)[0], :]
        population_data = sample_points[np.where(sample_points[:, 0] != user_id)[0], :]

        self._ind_kde = _build_adaptive_bandwidth_kde(user_data, nn, lon_to_km, lat_to_km)
        self._pop_kde = _build_adaptive_bandwidth_kde(population_data, nn, lon_to_km, lat_to_km)

    def log_pdf(self, query_point):
        """
        Log pdf of a point

         INPUT:
        -------
            1. query_point: (lon, lat) point to compute log pdf for

         OUTPUT:
        --------
            1. log_pdf: The log pdf for the query point
        """
        user_log_pdf = np.log(self._alpha) + self._ind_kde.log_pdf(query_point)
        pop_log_pdf = np.log(1 - self._alpha) + self._pop_kde.log_pdf(query_point)

        return np.logaddexp(user_log_pdf, pop_log_pdf)


class KDE(object):
    def __init__(self, data):
        """
        Returns a k-d tree like kde object.

         INPUT:
        -------
            1. data: The observed/train events.
                     The data is in the format of
                     np.array([[user_id, lon, lat, bandwidth, weigh], ... ])
                     where each line is a different point.
        """
        # There is no normalization at the end and the algorithm assumes the weights
        # in the train data are sum to 1. The regular notation assumes the weights sum
        # to N and that's why it is written as 1/N * sum ...
        data[:, -1] /= np.sum(data[:, -1])
        self.ball_tree = BallTree(data)

    def log_pdf(self, query_point):
        """
        Computed the estimated log pdf of a query point using a k-d tree
        approach for fast kde computation.

         INPUT:
        -------
            1. query_point: array like [lon, lat]

         OUTPUT:
        --------
            1. log pdf
        """
        return self.ball_tree.log_pdf(query_point)

    @staticmethod
    def sample_from_kde(data):
        """
        Sampling a point from data. This is done separately from the actual
        object because in the sampling I will usually change the weights all the
        time (because that's how the sampling is done) and I don't want to split
        the ball tree every time (it's expensive). Sampling from KDE is done in
        the following way:

            1. We sample a point from data according to the weights.
            2. We sample a point from MVN with the mean as the sampled point from
                (1) and the bw of the point from (1)


         INPUT:
        -------
            1. data: The observed/train events.
                    The data is in the format of
                    np.array([[user_id, lon, lat, bandwidth, weigh], ... ])
                    where each line is a different point.
                    NOTE: We assume the weights sum to 1!!!!

         OUTPUT:
        --------
            1. sampled_point: [lon, lat]
        """
        sampled_point_index = np.random.choice(data.shape[0], p=data[:, -1])
        sampled_point = data[sampled_point_index, :]

        sample_lon = np.random.normal(sampled_point[1], sampled_point[3] * KM_TO_LON)
        sample_lat = np.random.normal(sampled_point[2], sampled_point[3] * KM_TO_LAT)

        return [sample_lon, sample_lat]


class BallTree(object):
    def __init__(self, data):
        self.num_points = data.shape[0]
        try:
            self.head_ball = BallTree.split_ball(data, 0)
        except Exception as e:
            print("Creating ball tree with min parent %s".format(MIN_PARENT))
            print(e)

    def log_pdf(self, query_event):
        """
        Computes the log pdf for a query event by recursing on the tree.

         INPUT:
        -------
            1. query_event: (lon, lat)
                            The event to compute the log_pdf for

         OUTPUT:
        --------
            1. log_pdf: log pdf estimated value (float)
        """
        return self.log_pdf_recurse(self.head_ball, query_event)

    def log_pdf_recurse(self, ball, query_event):
        """
        The actual recursive function

         INPUT:
        -------
            1. ball:        The inspected current ball
            2. query_event: (lon, lat)
                            The event to compute the log_pdf for

         OUTPUT:
        --------
            1. log_pdf: log pdf estimated value (float)
        """
        if ball.ball_data is not None:
            # This means that we are in a ball with actual data in it
            return _log_kernel_for_ball_data(query_event, ball.ball_data)

        min_log_pdf = ball.min_log_pdf(query_event)
        max_log_pdf = ball.max_log_pdf(query_event)

        if np.exp(max_log_pdf) - np.exp(min_log_pdf) < TOL * self.num_points / ball.num_points:
            return np.log(ball.num_points) + np.logaddexp(max_log_pdf, min_log_pdf) - np.log(2)

        # Need to recurse
        return np.logaddexp(
            self.log_pdf_recurse(ball.left_ball, query_event),
            self.log_pdf_recurse(ball.right_ball, query_event),
        )

    @staticmethod
    def split_ball(data, feature):
        """
        """
        if data.shape[0] <= MIN_PARENT:
            return Ball(ball_data=data)

        lower_left = np.amin(data[:, 1:3], axis=0)
        upper_right = np.amax(data[:, 1:3], axis=0)
        max_bandwidth = np.amax(data[:, 3], axis=0)
        min_bandwidth = np.amin(data[:, 3], axis=0)
        max_weight = np.amax(data[:, 4], axis=0)
        min_weight = np.amin(data[:, 4], axis=0)

        # Finding the left and right ball. The median goes to the right ball
        arg_sort = np.argsort(
            data[:, feature + 1]
        )  # The '+1' is because the 0's column in the data is the user_id
        med_ind = int(np.ceil(len(arg_sort) / 2))
        left_ball_data = data[arg_sort[:med_ind], :]
        right_ball_data = data[arg_sort[med_ind:], :]

        left_ball = BallTree.split_ball(left_ball_data, (feature + 1) % 2)
        right_ball = BallTree.split_ball(right_ball_data, (feature + 1) % 2)

        return Ball(
            lower_left=lower_left,
            upper_right=upper_right,
            left_ball=left_ball,
            right_ball=right_ball,
            min_bw=min_bandwidth,
            max_bw=max_bandwidth,
            min_weight=min_weight,
            max_weight=max_weight,
            num_points=data.shape[0],
        )


class Ball(object):
    """
    The Ball class represent an area on the map that contains points.
    It is used in the kd - tree like search for the pdf.
    It can have two forms:
        1. Contains all the events in the area, in the case where the number
            of events is lower than the threshold
        2. Save just the mid event and some sufficient statistics that allows
            us to compute the pdf estimation
    """

    def __init__(self, **kwargs):
        """
        The constructor for the first case. Saving all the points

         INPUT:
        -------
            1. ball_data:   All the event in the area
        """
        self.ball_data = None
        if len(kwargs) == 1:
            self.ball_data = kwargs["ball_data"]
            self.num_points = len(self.ball_data)
        else:
            self.num_points = kwargs["num_points"]
            self.lower_left = kwargs["lower_left"]
            self.upper_right = kwargs["upper_right"]
            self.left_ball = kwargs["left_ball"]
            self.right_ball = kwargs["right_ball"]
            self.min_bw = kwargs["min_bw"]
            self.max_bw = kwargs["max_bw"]
            self.min_weight = kwargs["min_weight"]
            self.max_weight = kwargs["max_weight"]

    def min_log_pdf(self, query_point):
        """
        Returns the minimum log probability that can obtained from the ball

         INPUT:
        -------
            1. query_event: The query point [lon, lat]

         OUTPUT:
        --------
            1. log_pdf: The log probability
        """
        return _log_kernel_for_single_sample_event(
            query_point, self.farthest_point(query_point), self.min_bw, self.max_bw, self.max_weight
        )

    def max_log_pdf(self, query_point):
        """
        Returns the maximum log probability that can obtained from the ball

         INPUT:
        -------
            1. query_point: The query point [lon, lat]

         OUTPUT:
        --------
            1. log_pdf: The log probability
        """
        return _log_kernel_for_single_sample_event(
            query_point, self.closest_point(query_point), self.max_bw, self.min_bw, self.min_weight
        )

    def farthest_point(self, query_point):
        """
        Returns the farthest point in the box from the query point.
        The point is most likely not a real point.

         INPUT:
        -------
            1. query_point: The query point [lon, lat]

         OUTPUT:
        --------
            1. farthest_point: [lon, lat]
        """
        midy = self.upper_right[1] - (self.upper_right[1] - self.lower_left[1]) / 2
        midx = self.upper_right[0] - (self.upper_right[0] - self.lower_left[0]) / 2

        upper_left = [self.lower_left[0], self.upper_right[1]]
        lower_right = [self.upper_right[0], self.lower_left[1]]

        if query_point[0] > midx:
            if query_point[1] > midy:
                return self.lower_left

            return upper_left

        # Else - to the left ot the midx
        if query_point[1] > midy:
            return lower_right

        return self.upper_right

    def closest_point(self, query_point):
        """
        Returns the closest_point point in the ball from the query point
        The point is most likely not a real point.

         INPUT:
        -------
            1. query_point: The query point [lon, lat]

         OUTPUT:
        --------
            1. closest_point: [lon, lat]
        """
        if query_point[0] > self.upper_right[0]:
            # To the right of the area
            if query_point[1] >= self.upper_right[1]:
                return self.upper_right

            if query_point[1] >= self.lower_left[1]:
                return [self.upper_right[0], query_point[1]]

            return [self.upper_right[0], self.lower_left[1]]

        if query_point[0] > self.lower_left[0]:
            # between the two vertical edges (could be in the area)
            if query_point[1] > self.upper_right[1]:
                return [query_point[0], self.upper_right[1]]

            if query_point[1] >= self.lower_left[1]:
                # If it's in the area, the closest possible query_point is
                # the query_point itself
                return query_point

            return [query_point[0], self.lower_left[1]]

        # Else - to the left ot the area
        if query_point[1] > self.upper_right[1]:
            return [self.lower_left[0], self.upper_right[1]]

        if query_point[1] >= self.lower_left[1]:
            return [self.lower_left[0], query_point[1]]

        return self.lower_left


if __name__ == "__main__":
    dublin = np.genfromtxt("../data/dublin_filter_nobots.csv", delimiter=",")
    data = dublin[0:20, :3]
    data_kde = _build_adaptive_bandwidth_kde(data[:, 0:3])
    tmp = data_kde.log_pdf(data[1, 1:3])

    tmp = data_kde.log_pdf(data[-1, 1:3])
