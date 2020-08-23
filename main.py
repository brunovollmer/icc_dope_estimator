import sys, os
import argparse
import cv2
import numpy as np

from dope import DopeEstimator
from comparator import Comparator
from model import num_joints
from util import resize_image
from visualization import visualize_bodyhandface2d, visualize_differences, visualize_3d_pose

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
    parser.add_argument('--width', default=500, help='width of the visualization display')

    args = parser.parse_args()

    return args

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

        while(master_cap.isOpened()):
            master_ret, master_frame = master_cap.read()
            user_ret, user_frame = user_cap.read()

            if not master_ret or not user_ret:
                break

            master_frame = resize_image(master_frame, height=int(args.width))
            master_frame = cv2.cvtColor(master_frame, cv2.COLOR_BGR2RGB)

            master_results, master_res_img = dope.run(master_frame, visualize=args.visualize)

            user_frame = resize_image(user_frame, height=int(args.width))
            user_frame = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)

            user_results, user_res_img = dope.run(user_frame, visualize=args.visualize)

            differences = comparator.compare(master_results['body'][0]['pose3d'], user_results["body"][0]["pose3d"])

            user_result2d = {
                part: np.stack([d['pose2d'] for d in part_detections], axis=0)
                if len(part_detections) > 0
                else np.empty((0, num_joints[part], 2), dtype=np.float32)
                for part, part_detections in user_results.items()
            }

            user_res_img = visualize_differences(user_res_img, user_result2d, differences)

            if args.visualize:
                plot_image = visualize_3d_pose(user_results['body'][0]['pose3d'])
                plot_image = resize_image(plot_image, height=int(args.width))

                merg_res_img = cv2.hconcat([master_res_img, user_res_img, plot_image[...,:3]])

                cv2.imshow("result", merg_res_img)
                cv2.waitKey(0)


        master_cap.release()
        user_cap.release()
        cv2.destroyAllWindows()
