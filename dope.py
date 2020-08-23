import os
import sys
import numpy as np
import cv2
import torch
from torchvision.transforms import ToTensor
from PIL import Image

# add ppi repo to sys path and import it
curr_dir = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_dir, 'lcrnet-v2-improved-ppi'))

try:
    from lcr_net_ppi_improved import LCRNet_PPI_improved
except ModuleNotFoundError:
    raise Exception('To use the pose proposals integration (ppi) as postprocessing, please follow the readme instruction by cloning our modified version of LCRNet_v2.0 here. Alternatively, you can use --postprocess nms without any installation, with a slight decrease of performance.')

from model import dope_resnet50, num_joints
from postprocess import assign_hands_and_head_to_body
from visu import visualize_bodyhandface2d


class DopeEstimator:

    def __init__(self, model_path='models/DOPErealtime_v1_0_0.pth.tgz', use_half_comp=True):
        self._load_model(model_path, use_half_comp)


    def _load_model(self, model_path, use_half_comp):
        self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if not os.path.isfile(model_path):
            raise Exception(
                '{:s} does not exist. Please use correct model path.'.format(model_path))

        ckpt = torch.load(model_path)

        if not use_half_comp:
            ckpt['half'] = False

        ckpt['dope_kwargs']['rpn_post_nms_top_n_test'] = 1000

        model = dope_resnet50(**ckpt['dope_kwargs'])

        if ckpt['half']:
            model = model.half()

        model = model.eval()
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(self._device)

        self._model = model
        self._ckpt = ckpt

    def _post_process(self, results, filter_poses):

        parts = ['body', 'hand', 'face']

        res = {k: v.float().data.cpu().numpy() for k, v in results.items()}


        detections = {}
        for part in parts:
            detections[part] = LCRNet_PPI_improved(
                res[part+'_scores'], res['boxes'], res[part+'_pose2d'], res[part+'_pose3d'], self._resolution, **self._ckpt[part+'_ppi_kwargs'])

        # assignment of hands and head to body
        detections, hand_body, face_body = assign_hands_and_head_to_body(detections)

        if filter_poses:
            body_scores = [x['score'] for x in detections['body']]
            max_score_index = body_scores.index(max(body_scores))

            detections['body'] = [detections['body'][max_score_index]]
            detections['face'] = [detections['face'][face_body[max_score_index]]] if face_body[max_score_index] != -1 else []
            detections['hand'] = [detections['hand'][x] if x != -1 else [] for x in list(hand_body[max_score_index])]
            # # remove empty lists
            detections['hand'] = [x for x in detections['hand'] if x != []]

        return detections

    def _visualize_results(self, results, image):
        det_poses2d = {part: np.stack([d['pose2d'] for d in part_detections], axis=0) if len(part_detections) > 0 else np.empty(
            (0, num_joints[part], 2), dtype=np.float32) for part, part_detections in results.items()}

        scores = {part: [d['score'] for d in part_detections]
                  for part, part_detections in results.items()}

        res_img = visualize_bodyhandface2d(np.asarray(
            image)[:, :, ::-1], det_poses2d, dict_scores=scores)

        return res_img

    def run(self, image, visualize=False, filter_poses=True):
        # convert to PIL image
        image = Image.fromarray(image)

        tensor_list = [ToTensor()(image).to(self._device)]

        if self._ckpt['half']:
            tensor_list = [t.half() for t in tensor_list]

        self._resolution = tensor_list[0].size()[-2:]

        with torch.no_grad():
            results = self._model(tensor_list, None)[0]

        post_proc_results = self._post_process(results, filter_poses)

        if visualize:
            res_img = self._visualize_results(post_proc_results, image)
            return post_proc_results, res_img

        return post_proc_results, None
