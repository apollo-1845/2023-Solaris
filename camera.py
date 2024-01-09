from orbit import ISS
from exif import Image
from datetime import datetime


def convert(angle):
    """
    Convert a `skyfield` Angle to an Exif-appropriate representation (positive rationals),
    e.g. 98° 34' 58.7 to "98/1,34/1,587/10".

    :param angle: The angle to
    :return: A tuple containing a Boolean and the converted angle, with the boolean indicating if the angle is negative.
    """

    sign, degrees, minutes, seconds = angle.signed_dms()
    exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds * 10:.0f}/10'
    return sign < 0, exif_angle


def take_picture(camera, path):
    """
    Takes a picture from the raspberry pi, and saves it.
    :param camera: A PiCamera object that is used to take the picture.
    :param path: The path where the picture should be stored.
    :return: The picture taken.
    """
    south, longitude = convert(ISS().coordinates().longitude)
    west, latitude = convert(ISS().coordinates().latitude)

    camera.exif_tags['GPS.GPSLatitude'] = latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"

    camera.capture(path, format="jpg")

    return Image(path)


def get_time_difference(im1, im2):
    """
    Get the time difference between two images.
    :param im1: The first image, or image path.
    :param im2: The second image, or image path.
    :return: The time difference, in seconds.
    """
    if isinstance(im1, str):
        im1 = Image(im1)

    if isinstance(im2, str):
        im2 = Image(im2)

    im1_time = datetime.strptime(im1.get("datetime_original"), "%Y:%m:%d %H:%M:%S")
    im2_time = datetime.strptime(im2.get("datetime_original"), "%Y:%m:%d %H:%M:%S")

    return im2_time.second - im1_time.second
