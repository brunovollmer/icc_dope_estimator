import sys, os
import argparse
import cv2
import json
import numpy as np

from dope import DopeEstimator
from comparator import Comparator
from model import num_joints
from util import resize_image, save_json
from visualization import visualize_bodyhandface2d, visualize_differences, visualize_3d_pose
from skeleton import combine_poses

def parse_args():
    parser = argparse.ArgumentParser(description='running DOPE on an image: python dope.py --model <modelname> --image <imagename>')
    parser.add_argument('--model', required=True, type=str, help='name of the model to use (eg DOPE_v1_0_0)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='path to the image')
    group.add_argument('--video', type=str, help='path to video')
    parser.add_argument('--m_video', type=str, help='path to master video')
    parser.add_argument('--visualize', '-v', action='store_true', help='visualize results')
    parser.add_argument('--no_half_comp', action='store_true', help='disable half computation')
    parser.add_argument('--position_threshold', default=0.05)
    parser.add_argument('--angle_threshold', default=0.1)
    parser.add_argument('--width', default=1200, help='width of the visualization display')
    parser.add_argument('--save_images', action='store_true', help='save visualization results to results folder')
    parser.add_argument('--save_poses', action='store_true', help='save poses to results folder')

    args = parser.parse_args()

    return args


def make_img(master_frame, user_frame, master_pose, user_pose, corrected_pose):
    master_skeleton_img = visualize_3d_pose(master_pose)
    master_skeleton_img = resize_image(master_skeleton_img, height=400)

    user_skeleton_img = visualize_3d_pose(user_pose)
    user_skeleton_img = resize_image(user_skeleton_img, height=400)

    correct_skeleton_img = visualize_3d_pose(corrected_pose)
    correct_skeleton_img = resize_image(correct_skeleton_img, height=400)

    merg_skeleton_img = cv2.hconcat([
        master_skeleton_img,
        correct_skeleton_img,
        user_skeleton_img,
    ])
    merg_res_img = cv2.hconcat([
        master_frame,
        user_frame,
    ])

    merg_skeleton_img = resize_image(merg_skeleton_img, width=merg_res_img.shape[1])
    merg_img = cv2.vconcat([
        merg_res_img,
        merg_skeleton_img[..., :3]
    ])
    merg_img = resize_image(merg_img, width=args.width)

    return merg_img

if __name__=="__main__":

    args = parse_args()

    dope = DopeEstimator(args.model, not args.no_half_comp)
    comparator = Comparator(args.position_threshold, args.angle_threshold)

    if args.image:
        image = cv2.imread(args.image)
        image = resize_image(image, width=int(args.width))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = dope.run(image, visualize=args.visualize)

        cv2.destroyAllWindows()

    else:
        master_cap = cv2.VideoCapture(args.m_video)
        user_cap = cv2.VideoCapture(args.video)

        user_results = []
        master_results = []

        counter = 0

        while(master_cap.isOpened()):
            master_ret, master_frame = master_cap.read()
            user_ret, user_frame = user_cap.read()

            if not master_ret or not user_ret:
                break

            master_frame = resize_image(master_frame, height=int(args.width))
            master_frame = cv2.cvtColor(master_frame, cv2.COLOR_BGR2RGB)

            master_result, master_res_img = dope.run(master_frame, visualize=args.visualize)

            user_frame = resize_image(user_frame, height=int(args.width))
            user_frame = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)

            user_result, user_res_img = dope.run(user_frame, visualize=args.visualize)

            master_results.append(master_result)
            user_results.append(user_result)

            differences = comparator.compare(master_result['body'][0]['pose3d'], user_result["body"][0]["pose3d"])
            print("Frame", counter, differences)

            user_result2d = {
                part: np.stack([d['pose2d'] for d in part_detections], axis=0)
                if len(part_detections) > 0
                else np.empty((0, num_joints[part], 2), dtype=np.float32)
                for part, part_detections in user_result.items()
            }

            user_res_img = visualize_differences(user_res_img, user_result2d, differences)

            if args.visualize:
                user_pose3d = user_result["body"][0]["pose3d"]
                master_pose3d = master_result["body"][0]["pose3d"]
                corrected_pose = combine_poses(master_pose3d, user_pose3d, ([], [True for i in range(14)]))

                merg_img = make_img(master_res_img, user_res_img, master_pose3d, user_pose3d, corrected_pose)

                if args.save_images:
                    cv2.imwrite("results/{:03d}.jpg".format(counter), merg_img)

                cv2.imshow("result", merg_img)
                cv2.waitKey(1)

            counter += 1

        if args.save_poses:
            save_json(f'results/{args.video}_poses.json', user_results)
            save_json(f'results/{args.m_video}_poses.json', master_results)

        master_cap.release()
        user_cap.release()
        cv2.destroyAllWindows()
