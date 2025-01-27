import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dope_estimator.constants import *

def visualize_3d_pose(pose, joint_colors="black"):

    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    X_I = 0
    Y_I = 1
    Z_I = 2

    connections = [
        (FOOT_LEFT, KNEE_LEFT),
        (FOOT_RIGHT, KNEE_RIGHT),
        (KNEE_LEFT, HIP_LEFT),
        (KNEE_RIGHT, HIP_RIGHT),
        (HIP_LEFT, HIP_RIGHT),
        (HAND_LEFT, ELBOW_LEFT),
        (HAND_RIGHT, ELBOW_RIGHT),
        (ELBOW_LEFT, SHOULDER_LEFT),
        (ELBOW_RIGHT, SHOULDER_RIGHT),
        (SHOULDER_LEFT, SHOULDER_RIGHT),
        (HIP, NECK),
        (NECK, HEAD)
    ]

    rot_matrix_x = [
        [1,0,0],
        [0,np.cos(90), -np.sin(90)],
        [0,np.sin(90), np.cos(90)]
    ]

    rot_matrix_z = [
        [np.cos(180*np.pi/180), -np.sin(180*np.pi/180), 0],
        [np.sin(180*np.pi/180), np.cos(180*np.pi/180), 0],
        [0,0,1]
    ]

    pose = (rot_matrix_z @ pose.T).T

    #fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(-70,-90)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.set_axis_off()

    # add two extra points to pose
    if pose.shape[0] == 13:
        pose = np.append(pose, [(pose[HIP_LEFT]+pose[HIP_RIGHT])/2], axis=0)
        pose = np.append(pose, [(pose[SHOULDER_LEFT]+pose[SHOULDER_RIGHT])/2], axis=0)

    # plot all points
    xs = [x[X_I] for x in pose]
    ys = [x[Y_I] for x in pose]
    zs = [x[Z_I] for x in pose]

    ax.scatter(xs, ys, zs, c=joint_colors)


    # plot all lines

    for c in connections:
        x_l = [pose[c[0]][X_I],pose[c[1]][X_I]]
        y_l = [pose[c[0]][Y_I],pose[c[1]][Y_I]]
        z_l = [pose[c[0]][Z_I],pose[c[1]][Z_I]]

        ax.plot(x_l, y_l, z_l)


    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    #plt.show()
    canvas.draw()
    buffer = canvas.buffer_rgba()
    plot_image = np.asarray(buffer)

    return plot_image

def _get_bones_and_colors(J): # colors in BGR
    """
    param J: number of joints -- used to deduce the body part considered.
    """
    if J==13: # full body (similar to LCR-Net)
        lbones = [(9,11),(7,9),(1,3),(3,5)]
        rbones = [(0,2),(2,4),(8,10),(6,8)] + [(4,5),(10,11)] + [([4,5],[10,11]),(12,[10,11])]
        bonecolors = [ [0,255,0] ] * len(lbones) + [ [255,0,0] ] * len(rbones)
        pltcolors = [ 'g-' ] * len(lbones) + [ 'b-' ] * len(rbones)
        bones = lbones + rbones
    elif J==21: # hand (format similar to HO3D dataset)
        bones = [ [(0,n+1),(n+1,3*n+6),(3*n+6,3*n+7),(3*n+7,3*n+8)] for n in range(5)]
        bones = sum(bones,[])
        bonecolors = [(255,0,255)]*4 + [(255,0,0)]*4 + [(0,255,0)]*4 + [(0,255,255)]*4 + [(0,0,255)] *4
        pltcolors = ['m']*4 + ['b']*4 + ['g']*4 + ['y']*4 + ['r']*4
    elif J==84: # face (ibug format)
        bones = [ (n,n+1) for n in range(83) if n not in [32,37,42,46,51,57,63,75]] + [(52,57),(58,63),(64,75),(76,83)]
        # 32 x contour + 4 x r-sourcil +  4 x l-sourcil + 7 x nose + 5 x l-eye + 5 x r-eye +20 x lip + l-eye + r-eye + lip + lip
        bonecolors = 32 * [(255,0,0)] + 4*[(255,0,0)] + 4*[(255,255,0)] + 7*[(255,0,255)] + 5*[(0,255,255)] + 5*[(0,255,0)] + 18*[(0,0,255)] + [(0,255,255),(0,255,0),(0,0,255),(0,0,255)]
        pltcolors = 32  * ['b']       + 4*['b']       + 4*['c']         + 7*['m']         + 5*['y']         + 5*['g']       + 18*['r']       + ['y','g','r','r']
    else:
        raise NotImplementedError('unknown bones/colors for J='+str(J))
    return bones, bonecolors, pltcolors

def _get_xy(pose2d, i):
    if isinstance(i,int):
        return pose2d[i,:]
    else:
        return np.mean(pose2d[i,:], axis=0)

def _get_xy_tupleint(pose2d, i):
    return tuple(map(int,_get_xy(pose2d, i)))

def _get_xyz(pose3d, i):
    if isinstance(i,int):
        return pose3d[i,:]
    else:
        return np.mean(pose3d[i,:], axis=0)

def visualize_bodyhandface2d(im, dict_poses2d, dict_scores=None, lw=2, max_padding=100, bgr=True):
    """
    bgr: whether input/output is bgr or rgb

    dict_poses2d: some key/value among {'body': body_pose2d, 'hand': hand_pose2d, 'face': face_pose2d}
    """
    if all(v.size==0 for v in dict_poses2d.values()): return im

    h,w = im.shape[:2]
    bones = {}
    bonecolors = {}
    for k,v in dict_poses2d.items():
        bones[k], bonecolors[k], _ = _get_bones_and_colors(v.shape[1])

    # pad if necessary (if some joints are outside image boundaries)
    pad_top, pad_bot, pad_lft, pad_rgt = 0, 0, 0, 0
    for poses2d in dict_poses2d.values():
        if poses2d.size==0: continue
        xmin, ymin = np.min(poses2d.reshape(-1,2), axis=0)
        xmax, ymax = np.max(poses2d.reshape(-1,2), axis=0)
        pad_top = max(pad_top, min(max_padding, max(0, int(-ymin-5))))
        pad_bot = max(pad_bot, min(max_padding, max(0, int(ymax+5-h))))
        pad_lft = max(pad_lft, min(max_padding, max(0, int(-xmin-5))))
        pad_rgt = max(pad_rgt, min(max_padding, max(0, int(xmax+5-w))))

    imout = cv2.copyMakeBorder(im, top=pad_top, bottom=pad_bot, left=pad_lft, right=pad_rgt, borderType=cv2.BORDER_CONSTANT, value=[0,0,0] )
    if not bgr: imout = np.ascontiguousarray(imout[:,:,::-1])
    outposes2d = {}
    for part,poses2d in dict_poses2d.items():
        outposes2d[part] = poses2d.copy()
        outposes2d[part][:,:,0] += pad_lft
        outposes2d[part][:,:,1] += pad_top

    # for each part
    for part, poses2d in outposes2d.items():
        if part != "body": continue
        # draw each detection
        for ipose in range(poses2d.shape[0]): # bones
            pose2d = poses2d[ipose,...]

            # draw poses
            for ii, (i,j) in enumerate(bones[part]):
                p1 = _get_xy_tupleint(pose2d, i)
                p2 = _get_xy_tupleint(pose2d, j)
                cv2.line(imout, p1, p2, bonecolors[part][ii], thickness=lw*2)
            for j in range(pose2d.shape[0]):
                p = _get_xy_tupleint(pose2d, j)
                cv2.circle(imout, p, (2 if part!='face' else 1)*lw, (0,0,255), thickness=-1)

            # draw scores
            if dict_scores is not None: cv2.putText(imout, '{:.2f}'.format(dict_scores[part][ipose]), (int(pose2d[12,0]-10),int(pose2d[12,1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0) )

    if not bgr: imout = imout[:,:,::-1]

    return imout


joint_to_kpt = {
    0: 2,
    1: 3,
    2: 4,
    3: 5,
    4: 8,
    5: 9,
    6: 10,
    7: 11
}
def visualize_differences(image, pose, differences):

    for i, d in enumerate(differences[0]):
        if d:
            p = tuple(pose["body"][0, i])
            cv2.circle(image, p, 10, (255, 0, 0), thickness=2)
    for i, d, in enumerate(differences[1][:8]):
        if d:
            p = tuple(pose["body"][0, joint_to_kpt[i]])
            cv2.line(image, (int(p[0] - 7), int(p[1] - 7)), (int(p[0] + 7), int(p[1] + 7)), (0, 255, 0), thickness=2)
            cv2.line(image, (int(p[0] - 7), int(p[1] + 7)), (int(p[0] + 7), int(p[1] - 7)), (0, 255, 0), thickness=2)

    return image


