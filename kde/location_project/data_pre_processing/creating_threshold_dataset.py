"""
Author: Moshe Lichman
"""
from __future__ import division

import numpy as np
from location_project.utils import db_queries as db
import pickle


def finding_thresholded_users_from_dump_file(min_per_good_day, min_days, pickle_dump):
    """
    After dumping a very light thresholding users from the DB, this will load the dump and allow more crude testing.
    This way I can relatively fast can test different threshold params.


     INPUT:
    -------
        1. min_per_good_day:    The minimum amount of aggregated events in a day so the day will be consider "good day"
        2. min_days:            Minimum amount of days so the user will be consider as "good user"
        3. pickle_dump:         The dictionary that was created from the DB:
                                {user_id: {day_key: {aggregated_window_key: [list of the events]}}}

     OUTPUT:
    --------
       1. good_users_data: dictionary = {user_id: {day_key: {aggregated_window_key: [list of the events]}}}

            day_key:    (month, day)
            aggregated_window_key: (hour_of_the_day, minute of starting window)
            event:  (user_id, tweet_id, time_stamp, long, lat)
    """
    # So currently I was stupid enough to not save the uid's but it's still good for telling me how much users we have.
    good_users = 0 # change this to dictionary once it is fixed
    for user_data in pickle_dump:
        if len(user_data) < min_days:
            # This means not enough days
            continue

        # If we got here, then we need to check each day and make sure we have enough days with min number of tweets
        good_days = 0
        for day_data in user_data.values():
            if len(day_data) >= min_per_good_day:
                good_days += 1

            if good_days >= min_days:
                good_users += 1
                break

    print 'Number of good users with [%d %d] is: %d' % (min_per_good_day, min_days, good_users)


def finding_thresholded_users_from_db(min_per_good_day, min_days, agg_time_in_minutes, loc):
    """
    Finding users that comply with the threshold parameters. This is done by querying the DB so it's very slow.
    Therefore, it is done once to weed out the really low ones and pickle dump it.

    In the counting, we're actually looking at aggregated events in a time period. For example, if we want
    to aggregate over 10 minutes, it means that all the tweets between 1:00:00 and 1:09:59 will be counted as one.
    This is because some users tend to just use Twitter for conversation and we are only interested in their location
    every aggregated time.

    I ran it with:
        ok_users = finding_thresholded_users_from_db(5, 10, 20, 'DB') # for Dublin
        pickle.dump(ok_users, open('../data/users_5_10', 'w'))

     INPUT:
    -------
        1. min_per_good_day:    The minimum amount of aggregated events in a day so the day will be consider "good day"
        2. min_days:            Minimum amount of days so the user will be consider as "good user"
        3. agg_time_in_minutes: The window of time in minutes that we want to aggregate the data on.
        4. loc:                 Location to get data from

     OUTPUT:
    --------
       1. good_users_data: dictionary = {user_id: {day_key: {aggregated_window_key: [list of the events]}}}

            day_key:    (month, day)
            aggregated_window_key: (hour_of_the_day, minute of starting window)
            event:  (user_id, tweet_id, time_stamp, long, lat)
    """
    uids = db.get_unique_user_ids_from_raw_data(loc)
    print 'Got %d users from raw data' % len(uids)
    good_users_data = {}
    ind = 0
    for uid in uids:
        ind += 1
        if ind % 1000 == 0:
            print '%d users left - So far found %d good users' % (len(uids) - ind, len(good_users_data))

        user_raw_data = db.get_all_raw_data_for_user(uid,loc)

        # First collecting the aggregated data
        user_aggregated_data = {}
        for data_point in user_raw_data:
            dt = data_point[1]
            day_key = (dt.month, dt.day)
            aggregated_window_key = (dt.hour, np.floor(dt.minute / agg_time_in_minutes))

            # Sorry, I'm not a fan of default dict. They not work with pickle so I just got accustomed to not using them
            if day_key not in user_aggregated_data:
                user_aggregated_data[day_key] = {}

            if aggregated_window_key not in user_aggregated_data[day_key]:
                user_aggregated_data[day_key][aggregated_window_key] = []

            user_aggregated_data[day_key][aggregated_window_key].append(data_point)

        # Now making sure that the user follows the threshold
        if len(user_aggregated_data) < min_days:
            # Without even looking at the data in the days we can say that there's not enough data
            continue

        # Now let's look to make sure that there are enough "good days"
        good_days_count = 0
        for day_data in user_aggregated_data.values():
            if len(day_data) >= min_per_good_day:
                good_days_count += 1

        if good_days_count >= min_days:
            good_users_data[uid] = user_aggregated_data

    print 'Found %d good users' % len(good_users_data)
    return good_users_data

# if __name__ == '__main__':
#     ok_users = finding_thresholded_users_from_db(5, 10, 20)
#     pickle.dump(ok_users, open('/scratch/mlichman/location_data/users_5_10_20', 'w'))
#     print 'Done'

def dict_to_list(dict):
    """
    Converts dictionary returned from finding_thresholded_users_from_db() to a list for processing.

    INPUTS:
    --------
        1. dict: dictionary = {user_id: {day_key: {aggregated_window_key: [list of the events]}}}

    OUTPUT:
    -------
        1. good_users_list: array of thresholded data in format [user_id, lon, lat]
    """
    good_users_list = []
    for uid in dict:
        for day_key in dict[uid]:
            for agg_window_key in dict[uid][day_key]:
                for event in dict[uid][day_key][agg_window_key]:
                    tmp = [uid,event[-2],event[-1]]
                    good_users_list.append(tmp)

    return good_users_list

def find_bots(data):
    """
    Finds user_ids that could potentially be bots. Returns a list of the ID's.

    INPUTS:
    -------
        1. data: array of observed data in format np.array([[user_id, lon, lat, bandwidth], ... ])

    OUTPUT:
    -------
        1. bots: list of user_id that need to be investigated for being bots
    """
    bots = []
    uid = np.unique(data[:,0])
    for i in range(uid.shape[0]):
        ind = np.where(data[:,0] == uid[i])
        bw = data[ind[0],3]
        if (len(bw) >= 10 and sum(bw) == 0.0):
            bots.append(uid[i])

    return np.array(bots)

    # then filter tweets with the following:
    # no_bots = data[np.in1d(data[:,0], bots, invert=True), :]
