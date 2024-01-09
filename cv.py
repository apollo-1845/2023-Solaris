import math
import cv2


def convert_to_cv(path):
    """
    Convert an image to a ndarray.
    :param path: The path of the image.
    :return: The image as a ndarray.
    """
    return cv2.imread(path, 0)


def calculate_features(image, nfeatures):
    """
    Finds features of the image.
    :param image: The image.
    :param nfeatures: Maximum number of features to match.
    :return: A tuple consisting of the keypoints and descriptors.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def calculate_matches(descriptors_1, descriptors_2):
    """
    Find matching features.
    :param descriptors_1: The descriptors from the first image.
    :param descriptors_2: The descriptors from the second image.
    :return: The corresponding matches.
    """
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    """
    Find matching coordinates between two lists of keypoints, given the matches.
    :param keypoints_1: The first list of keypoints.
    :param keypoints_2: The second list of keypoints.
    :param matches: The matches.
    :return: A tuple consisting of the matching coordinates for each image.
    """
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1, y1) = keypoints_1[image_1_idx].pt
        (x2, y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1, y1))
        coordinates_2.append((x2, y2))
    return coordinates_1, coordinates_2


def calculate_mean_distance(coordinates_1, coordinates_2):
    """
    Calculate the mean distance between two images given the coordinates of features.
    :param coordinates_1: The feature coordinates from the first image.
    :param coordinates_2: The feature coordinates from the second image.
    :return: The mean distance, in pixels.
    """
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))

    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]

        distance = math.hypot(x_difference, y_difference)
        all_distances += distance

    return all_distances / len(merged_coordinates)


def calculate_speed(feature_distance, GSD, time_difference):
    """
    Calculate the speed of the ISS.
    :param feature_distance: The distance between features, in pixels.
    :param GSD: The Ground Sampling Distance, in cm/px.
    :param time_difference: The difference in the time that the images were taken.
    :return: The calculated speed of the ISS, in km/s.
    """
    CENTIMETERS_PER_KILOMETER = 100_000

    distance = feature_distance * GSD / CENTIMETERS_PER_KILOMETER
    speed = distance / time_difference
    return speed
