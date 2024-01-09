from datetime import timedelta

from picamera import PiCamera

from camera import *
from cv import *

N_FEATURES = 1000
GSD = 12648  # TODO: calculate if necessary
CAMERA_RESOLUTION = (4056, 3040)  # TODO: check if this is the right resolution.


def calculate_iss_speed(im1, im2):
    """
    Calculate the iss speed given two images.
    :param im1: The path to the first image.
    :param im2: The path to the second image.
    :return: The calculated speed, in km/s.
    """
    time_difference = get_time_difference(im1, im2)

    im1_cv = convert_to_cv(im1)
    im2_cv = convert_to_cv(im2)

    keypoints_1, descriptors_1 = calculate_features(im1_cv, N_FEATURES)
    keypoints_2, descriptors_2 = calculate_features(im2_cv, N_FEATURES)

    matches = calculate_matches(descriptors_1, descriptors_2)

    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
    mean_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)

    speed = calculate_speed(mean_feature_distance, GSD, time_difference)
    return speed


def write_result(speed_estimate):
    """
    Write the result to result.txt
    :param speed_estimate: The calculated speed of the iss, in km/s.
    """
    with open("result.txt", "w") as file:
        file.write(str(float('%.5g' % speed_estimate)))


def main():
    start_time = datetime.now()
    now_time = datetime.now()

    camera = PiCamera()

    print("Program starting")

    speeds = []

    prev_image_path = None

    i = 0

    # TODO: set the timedelta to the amount of time you want the program to take,
    #  minus the time for one iteration to complete.
    while now_time < start_time + timedelta(minutes=3):
        print(f"Iteration {i}")

        current_image_path = f"./image_{i}.jpg"
        take_picture(camera, current_image_path)

        if prev_image_path:
            speeds.append(calculate_iss_speed(prev_image_path, current_image_path))

        prev_image_path = current_image_path

        now_time = datetime.now()
        i += 1

    print("Capturing over")

    camera.close()
    result = sum(speeds) / len(speeds)
    print(f"Result: {result}")
    write_result(result)

    print("Program over")


if __name__ == '__main__':
    main()
