import numpy as np
from constants import *


"""
Scale pose so that spine has roughly length 1
"""
def normalize_skeleton(poses):
    scale = 0
    for pose in poses:
        scale += np.linalg.norm(pose[NECK] - pose[HIP])
    scale /= len(poses)

    poses /= scale
    return poses


"""
Make sequences same length
"""
def align_poses(master_poses, user_poses):
    length_diff = len(master_poses) - len(user_poses)
    if length_diff > 0:
        print(f"Repeat last user pose {length_diff} times to match length")
        user_poses += user_poses[-1] * length_diff
    elif length_diff < 0:
        print(f"Repeat last master pose {-length_diff} times to match length")
        master_poses += master_poses[-1] * -length_diff
    else:
        print("Lengths of sequences match")
    return master_poses, user_poses


def compare_poses(master_poses, user_poses):
    #master_poses = normalize_skeleton(master_poses)
    #user_poses = normalize_skeleton(user_poses)

    master_poses, user_poses = align_poses(master_poses, user_poses)

    pose_scores = []
    joint_scores = []
    for master_pose, user_pose in zip(master_poses, user_poses):
        pose_diff = master_pose - user_pose
        pose_diff -= pose_diff[ROOT_JOINT]
        pose_dist = np.linalg.norm(pose_diff, axis=1)

        scores = []
        for d in pose_dist:
            if d < 0.05:
                score = 0
            elif d < 0.1:
                score = 1
            else:
                score = 2
            scores.append(score)
        joint_scores.append(scores)

    return np.array(joint_scores)

if __name__ == "__main__":
    import json
    import cv2
    from argparse import ArgumentParser
    from visualization import visualize_3d_pose

    parser = ArgumentParser()
    parser.add_argument("master_poses")
    parser.add_argument("user_poses")

    args = parser.parse_args()

    def load_3d_poses(path):
        print(f"Loading {path}...")
        with open(path) as f:
            data = json.load(f)
        poses = [p["body"][0]["pose3d"] for p in data]
        return np.array(poses)

    master_poses = load_3d_poses(args.master_poses)
    user_poses = load_3d_poses(args.user_poses)

    if master_poses.shape[1] == 13:
        def add_hip_neck(poses):
            def hip_neck(poses):
                return (
                    (poses[:, HIP_LEFT, :] + poses[:, HIP_RIGHT, :]) / 2,
                    (poses[:, SHOULDER_LEFT, :] + poses[:, SHOULDER_RIGHT, :]) / 2,
                )
            hip, neck = hip_neck(poses)
            hip = np.expand_dims(hip, axis=1)
            neck = np.expand_dims(neck, axis=1)
            return np.concatenate((poses, hip, neck), axis=1)
        master_poses = add_hip_neck(master_poses)
        user_poses = add_hip_neck(user_poses)

    print(f"Loaded {len(master_poses)} master poses, {len(user_poses)} user poses")

    print(f"Computing scores...")
    scores = compare_poses(master_poses, user_poses)
    print(f"Computed {len(scores)} scores, worst score {scores.max()}, best score {scores.min()}")
    print("Average scores per joint:")
    avg_scores = np.average(scores, axis=0)
    print(avg_scores)
    print(scores.shape)
    print(scores)

    for m, u, s in zip(master_poses, user_poses, scores):
        master_img = visualize_3d_pose(m)
        user_img = visualize_3d_pose(u)
        res_img = cv2.hconcat([master_img, user_img])
        cv2.imshow("Pose difference", res_img)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
