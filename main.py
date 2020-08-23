import sys, os
import argparse
import cv2

from dope import DopeEstimator
from util import resize_image

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='running DOPE on an image: python dope.py --model <modelname> --image <imagename>')

    parser.add_argument('--model', required=True, type=str, help='name of the model to use (eg DOPE_v1_0_0)')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='path to the image')
    group.add_argument('--video', type=str, help='path to video')

    parser.add_argument('--visualize', '-v', action='store_true', help='visualize results')

    parser.add_argument('--no_half_comp', action='store_true', help='disable half computation')

    args = parser.parse_args()

    dope = DopeEstimator(args.model, not args.no_half_comp)

    if args.image:
        image = cv2.imread(args.image)
        image = resize_image(image, width=500)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = dope.run(image, visualize=args.visualize)

        cv2.destroyAllWindows()

    else:
        cap = cv2.VideoCapture(args.video)

        while(cap.isOpened()):
            ret, frame = cap.read()

            if not ret:
                break

            frame = resize_image(frame, width=1000)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = dope.run(frame, visualize=args.visualize)

        cap.release()
        cv2.destroyAllWindows()
