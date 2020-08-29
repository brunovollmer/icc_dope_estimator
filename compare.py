import numpy as np
from constants import *


THRESH_PERFECT = 0.02
THRESH_GOOD = 0.04
THRESH_OK = 0.08
THRESH_WRONG = 0.12

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
    master_poses = np.array(master_poses)
    user_poses = np.array(user_poses)

    if len(master_poses) == 0 or len(user_poses) == 0:
        return np.array([])

    master_poses, user_poses = align_poses(master_poses, user_poses)

    pose_scores = []
    joint_scores = []
    for master_pose, user_pose in zip(master_poses, user_poses):
        pose_diff = master_pose - user_pose
        pose_diff -= pose_diff[ROOT_JOINT]
        pose_dist = np.linalg.norm(pose_diff, axis=1)

        scores = []
        for d in pose_dist:
            if d < THRESH_PERFECT:
                score = 0
            elif d < THRESH_GOOD:
                score = 1
            elif d < THRESH_OK:
                score = 2
            elif d < THRESH_WRONG:
                score = 3
            else:
                score = 4
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
    for j in range(15):
        print(f"{avg_scores[j]:.4f}")

    def nop(x):
        pass

    num_frames = len(scores)
    STEPS = 1000

    cv2.namedWindow("image")
    cv2.createTrackbar("perfect", "image", int(THRESH_PERFECT * STEPS), STEPS, nop)
    cv2.createTrackbar("good", "image", int(THRESH_GOOD * STEPS), STEPS, nop)
    cv2.createTrackbar("ok", "image", int(THRESH_OK * STEPS), STEPS, nop)
    cv2.createTrackbar("wrong", "image", int(THRESH_WRONG * STEPS), STEPS, nop)

    cv2.createTrackbar("frame", "image", 0, num_frames, nop)
    score_colors = ["#00ff00", "#66dd00", "#999900", "#aa6600", "#ff0000"]

    while(1):
        THRESH_PERFECT = cv2.getTrackbarPos("perfect", "image") / STEPS
        THRESH_GOOD = cv2.getTrackbarPos("good", "image") / STEPS
        THRESH_OK = cv2.getTrackbarPos("ok", "image") / STEPS
        THRESH_WRONG = cv2.getTrackbarPos("wrong", "image") / STEPS
        cur_frame = cv2.getTrackbarPos("frame", "image")

        cur_master_pose = np.array([master_poses[cur_frame]])
        cur_user_pose = np.array([user_poses[cur_frame]])
        cur_scores = compare_poses(cur_master_pose, cur_user_pose)

        joint_colors = [score_colors[_s] for _s in cur_scores[0]]
        master_img = visualize_3d_pose(cur_master_pose[0])
        user_img = visualize_3d_pose(cur_user_pose[0], joint_colors=joint_colors)
        res_img = cv2.hconcat([master_img, user_img])
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", res_img[:, :, ::-1])

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    """
    for m, u, s in zip(master_poses, user_poses, scores):
        joint_colors = [score_colors[_s] for _s in s]
        master_img = visualize_3d_pose(m)
        user_img = visualize_3d_pose(u, joint_colors=joint_colors)
        res_img = cv2.hconcat([master_img, user_img])
        cv2.imshow("Pose difference", res_img)
        cv2.waitKey(10)
    """
    cv2.destroyAllWindows()
