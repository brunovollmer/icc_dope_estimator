import sys, os
import argparse
import cv2
import numpy as np

from dope import DopeEstimator
from comparator import Comparator
from util import resize_image
from visu import visualize_bodyhandface2d, visualize_differences

dummy_master_pose3d = np.array([[ 7.3582644e-04, -1.0215719e+00,  7.9771616e-02],
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
dummy_master_pose2d = np.array([[150.4483  , 134.84048 ],
       [154.93839 , 139.75493 ],
       [139.87584 , 107.21053 ],
       [147.67429 , 108.86596 ],
       [145.9606  ,  78.49403 ],
       [150.53584 ,  78.1335  ],
       [145.96315 ,  75.788826],
       [154.74434 ,  82.974434],
       [150.74808 ,  60.09311 ],
       [157.53069 ,  61.70961 ],
       [149.51952 ,  39.470844],
       [155.26875 ,  39.391006],
       [146.50296 ,  23.37331 ]])

num_joints = {'body': 13, 'hand': 21, 'face': 84}


def visualize_results(image, master2d, result2d, differences):
    res_img = visualize_bodyhandface2d(np.asarray(image)[:, :, ::-1], master2d)
    res_img = visualize_bodyhandface2d(np.asarray(res_img)[:, :, ::-1], result2d)
    res_img = visualize_differences(res_img, result2d, differences)

    cv2.imshow("result", res_img)

    cv2.waitKey(1)

if __name__=="__main__":
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

    dope = DopeEstimator(args.model, not args.no_half_comp)
    comparator = Comparator(args.position_threshold, args.angle_threshold)

    if args.image:
        image = cv2.imread(args.image)
        image = resize_image(image, width=args.width)

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

            master_frame = resize_image(master_frame, height=args.width)
            master_frame = cv2.cvtColor(master_frame, cv2.COLOR_BGR2RGB)

            master_results, master_res_img = dope.run(master_frame, visualize=args.visualize)

            user_frame = resize_image(user_frame, height=args.width)
            user_frame = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)

            user_results, user_res_img = dope.run(user_frame, visualize=args.visualize)


            if args.visualize:
                merg_res_img = cv2.hconcat([master_res_img, user_res_img])

                cv2.imshow("result", merg_res_img)
                cv2.waitKey(1)



            # differences = comparator.compare(dummy_master_pose3d, results["body"][0]["pose3d"])

            # if args.visualize:
            #     result2d = {
            #         part: np.stack([d['pose2d'] for d in part_detections], axis=0)
            #         if len(part_detections) > 0
            #         else np.empty((0, num_joints[part], 2), dtype=np.float32)
            #         for part, part_detections in results.items()
            #     }
            #     master2d = {
            #         "hand": np.empty((0, num_joints["hand"], 2), dtype=np.float32),
            #         "face": np.empty((0, num_joints["face"], 2), dtype=np.float32),
            #         "body": np.expand_dims(dummy_master_pose2d, axis=0)
            #     }
            #     visualize_results(frame, master2d, result2d, differences)

        master_cap.release()
        user_cap.release()
        cv2.destroyAllWindows()
