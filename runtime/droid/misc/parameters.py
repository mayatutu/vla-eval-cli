import os
from cv2 import aruco

# Robot Params #
nuc_ip = os.environ['VLA_NUC_IP']
robot_ip = os.environ.get('VLA_ROBOT_IP', '')
laptop_ip = os.environ.get('VLA_CLIENT_IP', '')
sudo_password = ""  # intentionally omitted from snapshot
robot_type = "fr3"  # 'panda' or 'fr3'
robot_serial_number = os.environ.get('VLA_ROBOT_SERIAL', '')

# Camera ID's #
hand_camera_id = os.environ['VLA_WRIST_CAMERA']
varied_camera_1_id = os.environ['VLA_RIGHT_CAMERA']
varied_camera_2_id = os.environ['VLA_LEFT_CAMERA']

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
if hasattr(aruco, "Dictionary_get"):
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)
else:
    ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

# Ubuntu Pro Token (RT PATCH) #
ubuntu_pro_token = ""  # intentionally omitted from snapshot

# Code Version [DONT CHANGE] #
droid_version = "1.3"
