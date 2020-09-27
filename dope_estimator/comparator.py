import numpy as np

from dope_estimator.constants import *

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
