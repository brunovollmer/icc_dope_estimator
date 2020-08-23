import numpy as np


FOOT_LEFT = 0
FOOT_RIGHT = 1
KNEE_LEFT = 2
KNEE_RIGHT = 3
HIP_LEFT = 4
HIP_RIGHT = 5
HAND_LEFT = 6
HAND_RIGHT = 7
ELBOW_LEFT = 8
ELBOW_RIGHT = 9
SHOULDER_LEFT = 10
SHOULDER_RIGHT = 11
HEAD = 12


def get_root_point(pose):
    mid_hip = pose[HIP_LEFT] - pose[HIP_RIGHT]
    root_point = pose[HEAD] - mid_hip
    return root_point


def get_angle(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    return np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def get_joint_angles(pose):
    joints = [
        [FOOT_LEFT, KNEE_LEFT, HIP_LEFT],
        [FOOT_RIGHT, KNEE_RIGHT, HIP_RIGHT],
        [KNEE_LEFT, HIP_LEFT, HIP_RIGHT],
        [KNEE_RIGHT, HIP_RIGHT, HIP_LEFT],
        [HAND_LEFT, ELBOW_LEFT, SHOULDER_LEFT],
        [HAND_RIGHT, ELBOW_RIGHT, SHOULDER_RIGHT],
        [ELBOW_LEFT, SHOULDER_LEFT, SHOULDER_RIGHT],
        [ELBOW_RIGHT, SHOULDER_RIGHT, SHOULDER_LEFT]
    ]
    points = [[pose[j[0]], pose[j[1]], pose[j[2]]] for j in joints]
    points += [
        [pose[HIP_LEFT], (pose[HIP_LEFT] + pose[HIP_RIGHT])/2, pose[HEAD]],
        [pose[SHOULDER_LEFT], (pose[SHOULDER_LEFT] + pose[SHOULDER_RIGHT])/2, pose[HEAD]]
    ]
    joint_angles = [get_angle(*j) for j in points]

    return np.array(joint_angles)


class Comparator:
    def __init__(self, position_threshold=0.01, angle_threshold=0.01):
        self.position_threshold = position_threshold
        self.angle_threshold = angle_threshold

    def compare(self, pose1, pose2):
        root1 = get_root_point(pose1)
        root2 = get_root_point(pose2)

        position_dist = np.linalg.norm((pose1 - root1) - (pose2 - root2), axis=1)
        angle_dist = np.abs(get_joint_angles(pose1) - get_joint_angles(pose2))

        return position_dist > self.position_threshold, angle_dist > self.angle_threshold



pose2 = np.array([[ 7.3582644e-04, -1.0215719e+00,  7.9771616e-02],
       [-6.4732976e-02, -1.0403967e+00, -9.1906205e-02],
       [ 1.5451697e-01, -6.4252871e-01,  4.1176986e-02],
       [ 3.9898686e-02, -6.4531237e-01, -1.2062486e-01],
       [ 5.3283609e-02, -2.5014526e-01,  7.8500099e-02],
       [-5.1617232e-04, -2.4716055e-01, -9.4915509e-02],
       [ 6.3154943e-02, -2.2897005e-01,  2.7811000e-01],
       [-6.9807261e-02, -2.3943152e-01, -2.7555224e-01],
       [-7.2693708e-03, -3.4201786e-02,  2.3415668e-01],
       [-9.7016327e-02, -2.9824348e-02, -2.1604063e-01],
       [ 1.7679764e-02,  2.3381509e-01,  1.7545928e-01],
       [-6.5085523e-02,  2.3031504e-01, -1.5488963e-01],
       [ 4.6184860e-02,  4.2873949e-01, -2.5871873e-03]])
