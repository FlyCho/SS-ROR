import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from statistics import mean
from module.stn import spatial_transformer_network as transformer

import locality_aware_nms as nms_locality
import lanms

tf.app.flags.DEFINE_integer('class_num', 230, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './checkpoints/', '')
tf.app.flags.DEFINE_string('output_dir', 'outputs/', '')
tf.app.flags.DEFINE_bool('no_write_images', True, 'do not write images')
# tf.app.flags.DEFINE_bool('use_vacab', True, 'strong, normal or weak')
from module import Backbone_branch, RoI_rotate, classification_branch
from data_provider.data_utils import restore_rectangle

FLAGS = tf.app.flags.FLAGS
detect_part = Backbone_branch.Backbone(is_training=False)
roi_rotate_part = RoI_rotate.RoIRotate()
recognize_part = classification_branch.Recognition(is_training=False)
font = cv2.FONT_HERSHEY_SIMPLEX

blister_pack_names_input_path = "./blister_class.txt"
blister_names = []
f = open(blister_pack_names_input_path, 'r')
for l in f:
    l = l.strip('\n')
    blister_names.append(l)
f.close()

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''

    input_size = 512
    new_h, new_w, _ = im.shape
    max_h_w_i = np.max([new_h, new_w, input_size])
    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
    im_padded[:new_h, :new_w, :] = im.copy()
    im = im_padded
    # resize the image to input size
    new_h, new_w, _ = im.shape
    resize_h = input_size
    resize_w = input_size
    im = cv2.resize(im, dsize=(resize_w, resize_h))
    resize_ratio_3_x = resize_w / float(new_w)
    resize_ratio_3_y = resize_h / float(new_h)

    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def get_project_matrix_and_width(text_polyses, target_height=16):
    project_matrixes = []
    box_widths = []
    filter_box_masks = []
    # max_width = 0
    # max_width = 0

    for i in range(text_polyses.shape[0]):
        x1, y1, x2, y2, x3, y3, x4, y4 = text_polyses[i] / 4

        rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
        box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

        if box_w <= box_h:
            box_w, box_h = box_h, box_w

        mapped_x1, mapped_y1 = (0, 0)
        mapped_x4, mapped_y4 = (0, target_height)
        mapped_x2, mapped_y2 = (target_height, 0)

        # width_box = math.ceil(8 * box_w / box_h)
        # width_box = int(min(width_box, 128)) # not to exceed feature map's width
        # width_box = int(min(width_box, 512)) # not to exceed feature map's width
        """
        if width_box > max_width: 
            max_width = width_box 
        """

        # mapped_x3, mapped_y3 = (width_box, 8)

        src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
        dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)])
        affine_matrix = cv2.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        affine_matrix = affine_matrix.flatten()

        # project_matrix = cv2.getPerspectiveTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        # project_matrix = project_matrix.flatten()[:8]

        project_matrixes.append(affine_matrix)
        width_box = target_height
        box_widths.append(width_box)

    project_matrixes = np.array(project_matrixes)
    box_widths = np.array(box_widths)

    return project_matrixes, box_widths


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def dst_cal(src_x, src_y, dst_x, dst_y):
    x = dst_x - src_x
    y = dst_y - src_y
    # 用math.sqrt（）求平方根
    len = math.sqrt((x ** 2) + (y ** 2))
    return len


def polygon_sort(poly):
    point_len = []
    for i in range(4):
        point_len.append(dst_cal(poly[i][0], poly[i][1], 512, 0))
    len_sort = sorted(range(len(point_len)), key=lambda k: point_len[k])
    up_right_point = len_sort[0]
    # up_right_point = point_len.index(min(point_len))
    if (up_right_point - 1) < 0:
        last_point_index = 3
    else:
        last_point_index = up_right_point - 1

    if (up_right_point + 1) > 3:
        next_point_index = 0
    else:
        next_point_index = up_right_point + 1

    last_point_dst = dst_cal(poly[up_right_point][0], poly[up_right_point][1], poly[last_point_index][0],
                             poly[last_point_index][1])
    next_point_dst = dst_cal(poly[up_right_point][0], poly[up_right_point][1], poly[next_point_index][0],
                             poly[next_point_index][1])
    if last_point_dst > next_point_dst:
        point_index = []
        for i in range(up_right_point, up_right_point + 4, 1):
            if i >= 4:
                point_index.append(i - 4)
            else:
                point_index.append(i)
        poly = poly[(point_index[0], point_index[1], point_index[2], point_index[3]), :]
    else:
        point_index = []
        # up_right_point += 1
        up_right_point = len_sort[1]
        if up_right_point == 4:
            up_right_point = 0
        for i in range(up_right_point, up_right_point + 4, 1):
            if i >= 4:
                point_index.append(i - 4)
            else:
                point_index.append(i)
        poly = poly[(point_index[0], point_index[1], point_index[2], point_index[3]), :]

    return poly


def bktree_search(bktree, pred_word, dist=5):
    return bktree.query(pred_word, dist)


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def point_change(point):
    n_point = int(((point + 1) * 512) / 2)
    return n_point


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    """
    if FLAGS.use_vacab and os.path.exists("./vocab.txt"):
        bk_tree = BKTree(levenshtein, list_words('./vocab.txt'))
        # bk_tree = bktree.Tree()
    """
    with tf.get_default_graph().as_default():

        # define the placehodler
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        input_feature_map = tf.placeholder(tf.float32, shape=[None, None, None, 32], name='input_feature_map')
        input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 6], name='input_transform_matrix')
        input_box_mask = []
        input_box_mask.append(tf.placeholder(tf.int32, shape=[None], name='input_box_masks_0'))
        input_box_widths = tf.placeholder(tf.int32, shape=[None], name='input_box_widths')

        # define the model
        # input_seq_len = input_box_widths[tf.argmax(input_box_widths, 0)] * tf.ones_like(input_box_widths)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        shared_feature, f_score, f_geometry = detect_part.model(input_images)
        pad_rois = roi_rotate_part.roi_rotate_tensor_pad(input_feature_map, input_transform_matrix, input_box_mask,
                                                         input_box_widths)
        recognition_logits = recognize_part.build_graph(pad_rois, input_box_widths, class_num=FLAGS.class_num)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            # print(model_path.output.op.name)
            # 保存图
            tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
            # 把图和参数结构一起
            freeze_graph.freeze_graph('./output_model/pb_model/model.pb', '', False, model_path, 'out',
                                      'save/restore_all', 'save/Const:0', './output_model/pb_model/frozen_model.pb',
                                      False, "")


if __name__ == '__main__':
    tf.app.run()