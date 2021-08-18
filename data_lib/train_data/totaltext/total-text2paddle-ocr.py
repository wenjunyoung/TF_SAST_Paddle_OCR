import argparse
import glob
import os
import os.path as osp
import re
import json
import cv2
import mmcv
import numpy as np
import scipy.io as scio
import yaml
from shapely.geometry import Polygon

from mmocr.utils import convert_annotations, drop_orientation, is_not_png
from functools import partial

def collect_files(img_dir, gt_dir, split):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir(str): The image directory
        gt_dir(str): The groundtruth directory
        split(str): The split of dataset. Namely: training or test
    Returns:
        files(list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    # note that we handle png and jpg only. Pls convert others such as gif to
    # jpg or png offline
    suffixes = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
    # suffixes = ['.png']

    imgs_list = []
    for suffix in suffixes:
        imgs_list.extend(glob.glob(osp.join(img_dir, '*' + suffix)))

    imgs_list = sorted(
        [drop_orientation(f) if is_not_png(f) else f for f in imgs_list])
    ann_list = sorted(
        [osp.join(gt_dir, gt_file) for gt_file in os.listdir(gt_dir)])

    files = list(zip(imgs_list, ann_list))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, split, nproc=1):
    """Collect the annotation information.

    Args:
        files(list): The list of tuples (image_file, groundtruth_file)
        nproc(int): The number of process to collect annotations
    Returns:
        images(list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(nproc, int)

    load_img_info_with_split = partial(load_img_info, split=split)

    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info_with_split, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info_with_split, files)

    return images


def get_contours_mat(gt_path):
    """Get the contours and words for each ground_truth mat file.

    Args:
        gt_path(str): The relative path of the ground_truth mat file
    Returns:
        contours(list[lists]): A list of lists of contours
        for the text instances
        words(list[list]): A list of lists of words (string)
        for the text instances
    """
    assert isinstance(gt_path, str)

    contours = []
    words = []
    data = scio.loadmat(gt_path)
    data_polygt = data['polygt']

    for i, lines in enumerate(data_polygt):
        X = np.array(lines[1])
        Y = np.array(lines[3])

        point_num = len(X[0])
        word = lines[4]
        if len(word) == 0:
            word = '???'
        else:
            word = word[0]

        if word == '#':
            word = '###'
            continue

        words.append(word)

        arr = np.concatenate([X, Y]).T
        contour = []
        for i in range(point_num):
            contour.append(arr[i][0])
            contour.append(arr[i][1])
        contours.append(np.asarray(contour))

    return contours, words


def load_mat_info(img_info, gt_file):
    """Load the information of one ground truth in .mat format.

    Args:
        img_info(dict): The dict of only the image information
        gt_file(str): The relative path of the ground_truth mat
        file for one image
    Returns:
        img_info(dict): The dict of the img and annotation information
    """
    assert isinstance(img_info, dict)
    assert isinstance(gt_file, str)

    contours, words = get_contours_mat(gt_file)
    anno_info = []
    for contour in contours:
        if contour.shape[0] == 2:
            continue
        category_id = 1
        coordinates = np.array(contour).reshape(-1, 2)
        polygon = Polygon(coordinates)
        iscrowd = 0

        area = polygon.area
        # convert to COCO style XYWH format
        min_x, min_y, max_x, max_y = polygon.bounds
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox,
            area=area,
            segmentation=[contour])
        anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def process_line(line, contours, words):
    """Get the contours and words by processing each line in the gt file.

    Args:
        line(str): The line in gt file containing annotation info
        contours(list[lists]): A list of lists of contours
        for the text instances
        words(list[list]): A list of lists of words (string)
        for the text instances
    Returns:
        contours(list[lists]): A list of lists of contours
        for the text instances
        words(list[list]): A list of lists of words (string)
        for the text instances
    """

    line = '{' + line.replace('[[', '[').replace(']]', ']') + '}'
    ann_dict = re.sub('([0-9]) +([0-9])', r'\1,\2', line)
    ann_dict = re.sub('([0-9]) +([ 0-9])', r'\1,\2', ann_dict)
    ann_dict = re.sub('([0-9]) -([0-9])', r'\1,-\2', ann_dict)
    ann_dict = ann_dict.replace("[u',']", "[u'#']")
    ann_dict = yaml.load(ann_dict)

    X = np.array([ann_dict['x']])
    Y = np.array([ann_dict['y']])

    if len(ann_dict['transcriptions']) == 0:
        word = '???'
    else:
        word = ann_dict['transcriptions'][0]
        if len(ann_dict['transcriptions']) > 1:
            for ann_word in ann_dict['transcriptions'][1:]:
                word += ',' + ann_word
        word = str(eval(word))
    words.append(word)

    point_num = len(X[0])

    arr = np.concatenate([X, Y]).T
    contour = []
    for i in range(point_num):
        contour.append(arr[i][0])
        contour.append(arr[i][1])
    contours.append(np.asarray(contour))

    return contours, words


def get_contours_txt(gt_path):
    """Get the contours and words for each ground_truth txt file.

    Args:
        gt_path(str): The relative path of the ground_truth mat file
    Returns:
        contours(list[lists]): A list of lists of contours
        for the text instances
        words(list[list]): A list of lists of words (string)
        for the text instances
    """
    assert isinstance(gt_path, str)

    contours = []
    words = []

    with open(gt_path, 'r') as f:
        tmp_line = ''
        for idx, line in enumerate(f):
            # print('idx:{}, line:{}'.format(idx, line))
            line = line.strip()
            # print('line:{}'.format(line))
            if idx == 0:
                tmp_line = line
                continue
            if not line.startswith('x:'):
                tmp_line += ' ' + line
                continue
            else:
                complete_line = tmp_line
                tmp_line = line
            contours, words = process_line(complete_line, contours, words)

        if tmp_line != '':
            contours, words = process_line(tmp_line, contours, words)

        for word in words:

            if word == '#':
                word = '###'
                continue

    return contours, words

'''
def load_txt_info(gt_file, img_info):
    """Load the information of one ground truth in .txt format.

    Args:
        img_info(dict): The dict of only the image information
        gt_file(str): The relative path of the ground_truth mat
        file for one image
    Returns:
        img_info(dict): The dict of the img and annotation information
    """

    contours, words = get_contours_txt(gt_file)
    anno_info = []
    for contour in contours:
        if contour.shape[0] == 2:
            continue
        category_id = 1
        coordinates = np.array(contour).reshape(-1, 2)
        polygon = Polygon(coordinates)
        iscrowd = 0

        area = polygon.area
        # convert to COCO style XYWH format
        min_x, min_y, max_x, max_y = polygon.bounds
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox,
            area=area,
            segmentation=[contour])
        anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info
'''
def load_txt_info(gt_file, split, image_file):
    """Load the information of one ground truth in .txt format.

    Args:
        img_info(dict): The dict of only the image information
        gt_file(str): The relative path of the ground_truth mat
        file for one image
    Returns:
        img_info(dict): The dict of the img and annotation information
    """
    result = []    
    result.append(image_file)

    contours, words = get_contours_txt(gt_file)
    # print('contours:{}, words:{}'.format(contours, words))
    anno_info = []
    for id, contour in enumerate(contours):
        if contour.shape[0] == 2:
            continue
        category_id = 1
        coordinates = np.array(contour).reshape(-1, 2)
        polygon = Polygon(coordinates)
        iscrowd = 0

        area = polygon.area
        # convert to COCO style XYWH format
        min_x, min_y, max_x, max_y = polygon.bounds        

        min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
        h, w = max_x - min_x, max_y - min_y
        # 左上角
        x1 = min_x
        y1 = min_y
        # 右上角
        x2 = x1 + w
        y2 = y1
        # 右下角
        x3 = x1 + w
        y3 = y2 + h
        # 左下角
        x4 = x1
        y4 = y1 + h
        bbox = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        anno = dict(
            transcription=words[id],
            points=bbox)
        anno_info.append(anno)

    result.append(anno_info)
    # print(result)
    with open('Total-text/'+split+'_total_text_label.txt', 'a+') as f:
                    f.write(result[0]+'\t')
                    f.write(json.dumps(result[1])+'\n')
                    f.close()
    return result


def load_png_info(gt_file, img_info):
    """Load the information of one ground truth in .png format.

    Args:
        gt_file(str): The relative path of the ground_truth file for one image
        img_info(dict): The dict of only the image information
    Returns:
        img_info(dict): The dict of the img and annotation information
    """
    assert isinstance(gt_file, str)
    assert isinstance(img_info, dict)
    gt_img = cv2.imread(gt_file, 0)
    contours, _ = cv2.findContours(gt_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    anno_info = []
    for contour in contours:
        if contour.shape[0] == 2:
            continue
        category_id = 1
        xy = np.array(contour).flatten().tolist()

        coordinates = np.array(contour).reshape(-1, 2)
        polygon = Polygon(coordinates)
        iscrowd = 0

        area = polygon.area
        # convert to COCO style XYWH format
        min_x, min_y, max_x, max_y = polygon.bounds
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox,
            area=area,
            segmentation=[xy])
        anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def load_img_info(files, split):
    """Load the information of one image.

    Args:
        files(tuple): The tuple of (img_file, groundtruth_file)
    Returns:
        img_info(dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)

    img_file, gt_file = files
    # read imgs with ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')
    # read imgs with orientations as dataloader does when training and testing
    img_color = mmcv.imread(img_file, 'color')
    # make sure imgs have no orientation info, or annotation gt is wrong.
    assert img.shape[0:2] == img_color.shape[0:2]

    split_name = osp.basename(osp.dirname(img_file))

    image_file = osp.join(split_name, osp.basename(img_file))
    
    if osp.splitext(gt_file)[1] == '.mat':
        img_info = load_mat_info(img_info, gt_file)
    elif osp.splitext(gt_file)[1] == '.txt':
        img_info = load_txt_info(gt_file, split, image_file)
    else:
        raise NotImplementedError

    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert totaltext annotations to COCO format')
    parser.add_argument('root_path', help='totaltext root path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--split-list',
        nargs='+',
        help='a list of splits. e.g., "--split_list training test"')

    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    out_dir = args.out_dir if args.out_dir else root_path
    mmcv.mkdir_or_exist(out_dir)

    img_dir = osp.join(root_path, 'imgs')
    gt_dir = osp.join(root_path, 'annotations')

    set_name = {}
    for split in args.split_list:
        set_name.update({split: 'instances_' + split + '.json'})
        assert osp.exists(osp.join(img_dir, split))

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It takes {}s to convert totaltext annotation'):

            # ('./Total-text/imgs/test/img995.jpg', './Total-text/annotations/test/poly_gt_img994.txt')...
            files = collect_files(
                osp.join(img_dir, split), osp.join(gt_dir, split), split)
            
            image_infos = collect_annotations(files, split, nproc=args.nproc)
            # print(image_infos)
            # convert_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
