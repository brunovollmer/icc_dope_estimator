import numpy as np
import cv2

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
HIP = 13
NECK = 14

ROOT_JOINT = NECK


def get_rotation_matrix(v1, v2):
    rotation_matrix = np.eye(3)
    c = np.dot(v1, v2)
    if np.abs(c) > (1 - 1e-8):
        if c > 0:
            return rotation_matrix
        else:
            return -rotation_matrix
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix += kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def _4x4(m):
    new_m = np.eye(4)
    new_m[:3, :3] = m
    return new_m

def get_translation_matrix(t):
    res = np.eye(4)
    res[:3, 3] = t
    res[3, 3] = 1
    return res

class Skeleton:
    def __init__(self, keypoints):
        self._compute_skeleton(keypoints)

    def _compute_skeleton(self, keypoints):
        neckhip = np.array([
            (keypoints[HIP_LEFT] + keypoints[HIP_RIGHT]) / 2,
            (keypoints[SHOULDER_LEFT] + keypoints[SHOULDER_RIGHT]) / 2
        ])
        keypoints = np.vstack((keypoints, neckhip))
        self.keypoints = keypoints
        self.skeleton = [2, 3, 4, 5, 13, 13, 8, 9, 10, 11, 14, 14, 14, 14, 14]

        self.rotations = list(range(len(self.keypoints)))
        self.translations = list(range(len(self.keypoints)))
        self.transforms = list(range(len(self.keypoints)))
        self.world_transforms = list(range(len(self.keypoints)))

        shoulder = keypoints[SHOULDER_RIGHT] - keypoints[SHOULDER_LEFT]
        spine = keypoints[HEAD] - keypoints[HIP]
        normal = np.cross(spine, shoulder)
        normal = normal / np.linalg.norm(normal)

        camera = np.array([0, 0, 1])

        rotation = get_rotation_matrix(camera, normal)
        translation = get_translation_matrix(keypoints[NECK])

        self.root_transform = translation @ _4x4(rotation)

        self.rotations[ROOT_JOINT] = rotation
        self.translations[ROOT_JOINT] = translation
        self.transforms[ROOT_JOINT] = translation @ _4x4(rotation)
        self.world_transforms[ROOT_JOINT] = self.root_transform
        for joint in reversed(range(len(self.keypoints))):
            if joint == ROOT_JOINT:
                continue
            self._compute_transform(joint, self.skeleton[joint])

    def _compute_transform(self, joint1, joint2):
        translation = -(self.keypoints[joint2] - self.keypoints[joint1])
        translation = np.linalg.inv(self.world_transforms[joint2]) @ np.pad(translation, (0,1))
        translation = translation[:3]
        v = translation / np.linalg.norm(translation)
        rotation = get_rotation_matrix(np.array([0, 0, 1]), v)
        translation = get_translation_matrix(translation)

        self.translations[joint1] = translation
        self.rotations[joint1] = rotation
        self.transforms[joint1] = translation @ _4x4(rotation)
        self.world_transforms[joint1] = self.world_transforms[joint2] @ self.transforms[joint1]
        return None


    def get_pose(self, rotations):
        pose = np.zeros((len(self.keypoints), 4))
        pose[:, -1] = 1
        world_transforms = list(range(len(self.keypoints)))
        world_transforms[ROOT_JOINT] = np.eye(4)#self.root_transform
        for joint in reversed(range(len(self.keypoints))):
            prev_joint = self.skeleton[joint]
            if joint in rotations:
                rotation = rotations[joint]
            else:
                rotation = self.rotations[joint]
            world_transforms[joint] = world_transforms[prev_joint] @ self.translations[joint] @ _4x4(rotation)
            point = world_transforms[joint][:, 3]
            pose[joint] = point
        pose = pose[:, :3]
        return pose

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

pose = 0.3 * np.array([
    [0.8, 0.1, 1],
    [-0.5, 0.1, 1],
    [0.5, 1, 1],
    [-0.5, 1, 1],
    [0.5, 2, 1],
    [-0.5, 2, 1],
    [1, 2.2, 1],
    [-1, 2.2, 1.5],
    [1, 3, 1],
    [-1, 3, 1],
    [1, 4, 1],
    [-1, 4, 1],
    [0, 4.5, 1]
])
a = np.pi/2
rot = np.array([
    [np.cos(a), 0, -np.sin(a)],
    [0, 1, 0],
    [np.sin(a), 0 , np.cos(a)]
])
rot2 = np.array([
    [np.cos(a), -np.sin(a), 0],
    [np.sin(a), np.cos(a), 0],
    [0, 0, 1],
])
#pose = (rot @ pose.T).T
pose = np.array([[ 7.3582644e-04, -1.0215719e+00,  7.9771616e-02],
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

if __name__ == "__main__":
    from visualization import visualize_3d_pose
    skeleton = Skeleton(pose)
    new_pose = skeleton.get_pose({HIP_RIGHT: rot2})

    img = visualize_3d_pose(pose)
    img2 = visualize_3d_pose(new_pose)
    final_img = cv2.hconcat([img, img2])
    cv2.imshow("test", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

