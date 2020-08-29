import numpy as np
from constants import *


THRESH_PERFECT = 0.12
THRESH_GOOD = 0.2
THRESH_OK = 0.24
THRESH_WRONG = 0.0
DEFAULT_THRESHOLDS = {
    "perfect": THRESH_PERFECT,
    "good": THRESH_GOOD,
    "ok": THRESH_OK,
    "wrong": THRESH_WRONG
}

PEN_PERFECT = 0
PEN_GOOD = 1
PEN_OK = 2
PEN_WRONG = 3
PEN_SHIT = 4
PEN_NO_POSE = 0

OFFSET_PERC = 0.1

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
def align_poses(master_poses, user_poses, fill_zero=True):
    length_diff = len(master_poses) - len(user_poses)

    if length_diff > 0:

        if fill_zero:
            print(f"Add empty pose {length_diff} times to match length")
            user_poses = np.append(user_poses, np.zeros((length_diff, user_poses[0].shape[0], user_poses[0].shape[1])), axis=0)
        else:
            print(f"Repeat last user pose {length_diff} times to match length")
            user_poses += user_poses[-1] * length_diff
    elif length_diff < 0:
        length_diff = abs(length_diff)

        if fill_zero:
            print(f"Add empty pose {length_diff} times to match length")
            master_poses = np.append(master_poses, np.zeros((length_diff, master_poses[0].shape[0], master_poses[0].shape[1])), axis=0)
        else:
            print(f"Repeat last master pose {-length_diff} times to match length")
            master_poses += master_poses[-1] * -length_diff
    else:
        print("Lengths of sequences match")
    return master_poses, user_poses

def shift_pose(pose, offset):
    tmp_pose = np.zeros_like(pose)

    if offset > 0:
        tmp_pose[offset:] = pose[:(pose.shape[0] - offset)]

    elif offset < 0:
        offset = abs(offset)
        tmp_pose[:(pose.shape[0] - offset)] = pose[offset:]

    if offset == 0:
        return pose

    return tmp_pose

def compare_poses(master_poses, user_poses, thresholds, offset=0):
    user_poses = shift_pose(user_poses, offset)

    joint_scores = []
    for master_pose, user_pose in zip(master_poses, user_poses):

        if np.sum(master_pose) != 0 and np.sum(user_pose) != 0:
            pose_diff = master_pose - user_pose
            pose_diff -= pose_diff[ROOT_JOINT]
            pose_dist = np.linalg.norm(pose_diff, axis=1)

            scores = []
            for d in pose_dist:
                if d < thresholds['perfect']:
                    score = PEN_PERFECT
                elif d < thresholds['good']:
                    score = PEN_GOOD
                elif d < thresholds['ok']:
                    score = PEN_OK
                elif d < thresholds['wrong']:
                    score = PEN_WRONG
                else:
                    score = PEN_SHIT

                scores.append(score)

            joint_scores.append(scores)

        else:
            joint_scores.append([-1] * 15)

    return np.array(joint_scores)

def find_ideal_offset(master_poses, user_poses, timeshift_percentage=OFFSET_PERC, thresholds=DEFAULT_THRESHOLDS):
    master_poses, user_poses = align_poses(master_poses, user_poses)

    frames = master_poses.shape[0]

    offset_list = list(range(int(-frames*timeshift_percentage), int(frames*timeshift_percentage)))
    offset_results = []
    offset_scores = []

    for o in offset_list:
        offset_result = compare_poses(master_poses, user_poses, thresholds, offset=o)

        offset_results.append(offset_result)
        offset_scores.append(np.average(offset_result[offset_result >= 0]))

    best_offset_index = offset_scores.index(min(offset_scores))
    best_offset = offset_list[best_offset_index]
    best_result = offset_results[best_offset_index]

    print("best offset: {}".format(best_offset))

    user_poses = shift_pose(user_poses, best_offset)

    return best_result, best_offset, master_poses, user_poses

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

        poses = np.zeros((len(data), 15, 3))
        for i, p in enumerate(data):
            if p['body']:
                poses[i] = np.array(p['body'][0]['pose3d'])

        return poses

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

    scores, upd_master_poses, upd_user_poses = find_ideal_offset(master_poses, user_poses)


    print(f"Computed {len(scores)} scores, worst score {scores.max()}, best score {scores.min()}")
    print("Average scores per joint:")
    avg_scores = np.average(scores, axis=0)
    # for j in range(15):
    #     print(f"{avg_scores[j]:.4f}")

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
    score_colors = ["green", "blue", "yellow", "orange", "red"]

    while(1):
        THRESH_PERFECT = cv2.getTrackbarPos("perfect", "image") / STEPS
        THRESH_GOOD = cv2.getTrackbarPos("good", "image") / STEPS
        THRESH_OK = cv2.getTrackbarPos("ok", "image") / STEPS
        THRESH_WRONG = cv2.getTrackbarPos("wrong", "image") / STEPS
        cur_frame = cv2.getTrackbarPos("frame", "image")

        cur_master_pose = np.array([master_poses[cur_frame]])
        cur_user_pose = np.array([user_poses[cur_frame]])
        cur_scores = compare_poses(cur_master_pose, cur_user_pose, DEFAULT_THRESHOLDS)

        joint_colors = [score_colors[_s] for _s in cur_scores[0]]

        if np.sum(cur_master_pose[0]) != 0:
            master_img = visualize_3d_pose(cur_master_pose[0])
        else:
            master_img = np.zeros((400,500,4))

        if np.sum(cur_user_pose[0]) != 0:
            user_img = visualize_3d_pose(cur_user_pose[0], joint_colors=joint_colors)
        else:
            user_img = np.zeros((400,500,4))

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
