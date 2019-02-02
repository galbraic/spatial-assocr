"""
@author: Moshe Lichman

Module that contains different static functions that are used in Spatial analysis of data.
"""
from __future__ import division     # So we can have floating point division
import numpy as np

EARTH_RADIUS_IN_KM = 6371  # in KM - from the wiki page


def earth_dist_in_meters(point1, point2):
    """
    Computes the earth (sphere) distance (in meters) between point1 and point2
    Taken from http://www.johndcook.com/blog/python_longitude_latitude/

    Two edits:
        - It returns the distance in meters instead of the arc.
        - Added IF block for precision issues that happens (|cosine gets value| > 1 by .000000002)

    INPUT:
    ------
        1. point1:  np.array([lon,lat])
        2. points:  np.array[[lon_1,lat_1], [lon_2,lat_2], ...]

    OUTPUT:
        1. <float> Distance in meters
    """
    # Convert latitude and longitude to
    # spherical coordinates in radians.

    # phi = 90 - latitude
    phi1 = np.radians(90.0 - point1[1])
    phi2 = np.radians(90.0 - point2[1])

    # theta = longitude
    theta1 = np.radians(point1[0])
    theta2 = np.radians(point2[0])

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) +
           np.cos(phi1) * np.cos(phi2))

    if cos > 1:
        cos = 1
    elif cos < -1:
        cos = -1

    # This happens when for precision issue the cos is greater than 1 by a small fraction
    arc = np.arccos(cos)

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc * EARTH_RADIUS_IN_KM * 1000


def earth_dist_in_meters_for_multiple_points(point1, points):
    """
    Computes the earth (sphere) distance (in meters) between point1 and point2
    Taken from http://www.johndcook.com/blog/python_longitude_latitude/

    Two edits:
        - It returns the distance in meters instead of the arc.
        - Added IF block for precision issues that happens (|cosine gets value| > 1 by .000000002)

    INPUT:
    ------
        1. point1:  np.array([lon,lat])
        2. points:  np.array[[lon_1,lat_1], [lon_2,lat_2], ...]

    OUTPUT:
        1. <float> Distance in meters
    """
    # Convert latitude and longitude to
    # spherical coordinates in radians.

    # phi = 90 - latitude
    phi1 = np.radians(90.0 - point1[1])
    phi2 = np.radians(90.0 - points[:, 1])

    # theta = longitude
    theta1 = np.radians(point1[0])
    theta2 = np.radians(points[:, 0])

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) +
           np.cos(phi1) * np.cos(phi2))

    if cos > 1:
        cos = 1
    elif cos < -1:
        cos = -1

    # This happens when for precision issue the cos is greater than 1 by a small fraction
    arc = np.arccos(cos)

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc * EARTH_RADIUS_IN_KM * 1000

def angle_of_a_line_radians(point1, point2):
    """
    Computes the angle between the line starting at point1 and ends at point 2 compared to north going clockwise.
        (Similar to http://postgis.net/docs/manual-2.0/ST_Azimuth.html)

    That means that the angle of (0,0) and (0,3) will be 0. The input can be list, tuple np.array or anything
    that allows access the values as point[0] and point[1]

     INPUT:
    -------
        1. point1: Starting point of the line
        2. point2: End point of the line

     OUTPUT:
    --------
        1. angle: The angle between the line (clockwise) in radians (0 <= angle <= 2 * math.pi)
    """
    x = point2[0] - point1[0]
    y = point2[1] - point1[1]

    if x == y == 0: return 0

    omega = np.atan2(x,y)
    if omega < 0: # The other direction, making it clockwise
        omega += np.pi * 2

    return omega


def angle_of_a_line(point1, point2):
    """
    Computes the angle between the line starting at point1 and ends at point 2 compared to north going clockwise.
        (Similar to http://postgis.net/docs/manual-2.0/ST_Azimuth.html)

    That means that the angle of (0,0) and (0,3) will be 0. The input can be list, tuple np.array or anything
    that allows access the values as point[0] and point[1]

     INPUT:
    -------
        1. point1: Starting point of the line
        2. point2: End point of the line

     OUTPUT:
    --------
        1. angle: The angle between the line and north in degrees
    """
    x = point2[0] - point1[0]
    y = point2[1] - point1[1]

    if x == y == 0: return 0

    omega = np.degrees(np.atan(x/ y))

    # Fixing according to the quadrant:
    if x == 0 and y > 0: return 0
    if x == 0 and y < 0: return 180
    if x > 0 and y == 0: return 90
    if x < 0 and y == 0: return 270
    if x > 0 and y > 0: return omega
    if x > 0 and y < 0: return 180 + omega
    if x < 0 and y > 0: return 360 + omega
    if x < 0 and y < 0: return 180 + omega