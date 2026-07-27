import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
import os

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

def var_all_case(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, dataset_name="LA"):
    if dataset_name == "Pancreas_CT":
        with open('../data/Pancreas_CT/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["../data/Pancreas_CT/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]
    elif dataset_name == "BraTS2019":
        with open('../data/BraTS2019/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = ["../data/BraTS2019/data/" + item.replace('\n', '') + ".h5" for item in image_list]

    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = var_single_case_first_output(
            model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        
        if num_classes == 2:
            if np.sum(prediction) == 0:
                dice = 0
            else:
                dice = metric.binary.dc(prediction, label)
        else:
            dice_list = []
            for c in range(1, num_classes):
                pred_c = (prediction == c).astype(np.uint8)
                gt_c   = (label == c).astype(np.uint8)
                if gt_c.sum() == 0 and pred_c.sum() == 0:
                    dice_list.append(1.0)
                elif gt_c.sum() == 0 or pred_c.sum() == 0:
                    dice_list.append(0.0)
                else:
                    dice_list.append(metric.binary.dc(pred_c, gt_c))
            dice = np.mean(dice_list)
        
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_all_case(model_name, num_outputs, model_1, model_2, image_list, num_classes,
                  patch_size=(112,112,80), stride_xy=18, stride_z=4,
                  save_result=True, test_save_path=None, preproc_fn=None,
                  metric_detail=1, nms=0):
    
    ith = 0
    if num_classes == 2:
        total_metric = np.zeros(4)
    else:
        total_metric = np.zeros((num_classes - 1, 4))
    detail_logs = []

    for image_path in image_list:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_first_output(
            model_1, model_2, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        
        if nms and num_classes == 2:
            prediction = getLargestCC(prediction)

        if num_classes == 2:
            if np.sum(prediction) == 0:
                single_metric = (0, 0, 0, 0)
            else:
                single_metric = calculate_metric_percase(prediction, label[:])
            if metric_detail:
                line = '%02d, %.5f, %.5f, %.5f, %.5f' % (ith, *single_metric)
                print(line); detail_logs.append(line)
            total_metric += np.asarray(single_metric)
        else:
            single_metric = calculate_metric_multiclass(prediction, label[:], num_classes)
            if metric_detail:
                for c_idx in range(num_classes - 1):
                    line = 'case %02d, class %d, %.5f, %.5f, %.5f, %.5f' % \
                           (ith, c_idx + 1, *single_metric[c_idx])
                    print(line); detail_logs.append(line)
                mean_line = 'case %02d, mean, %s' % (ith, ', '.join(
                    ['%.5f' % v for v in single_metric.mean(axis=0)]))
                print(mean_line); detail_logs.append(mean_line)
            total_metric += single_metric

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_pred_1.nii.gz" % ith)
            nib.save(nib.Nifti1Image(score_map[1].astype(np.float32) if num_classes >= 2 
                                     else score_map[0].astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_scores.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label.astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    
    if num_classes == 2:
        print('average metric is {}'.format(avg_metric))
    else:
        for c_idx in range(num_classes - 1):
            print('class %d avg metric: %s' % (c_idx + 1, avg_metric[c_idx]))
        print('mean over classes: %s' % avg_metric.mean(axis=0))

    detail_log_file = os.path.join(test_save_path, f'../{model_name}_performance.csv')
    with open(detail_log_file, 'w') as f:
        f.write('\n'.join(detail_logs))
        f.write('\nAverage metric: {}\n'.format(avg_metric))
        if num_classes > 2:
            f.write('Mean over classes: {}\n'.format(avg_metric.mean(axis=0)))

    return avg_metric

def var_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=2):
    w, h, d = image.shape

    add_pad = False
    w_pad = max(patch_size[0] - w, 0); add_pad = add_pad or (w_pad > 0)
    h_pad = max(patch_size[1] - h, 0); add_pad = add_pad or (h_pad > 0)
    d_pad = max(patch_size[2] - d, 0); add_pad = add_pad or (d_pad > 0)
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
                       mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y_ in range(sy):
            ys = min(stride_xy * y_, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, 0), 0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y = model(test_patch)
                    if isinstance(y, (tuple, list)):
                        y = y[0]
                    y = F.softmax(y, dim=1)
                
                y = y.cpu().data.numpy()[0]
                
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    
    if num_classes == 2:
        label_map = (score_map[1] > 0.5).astype(np.uint8)
    else:
        label_map = np.argmax(score_map, axis=0).astype(np.uint8)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def test_single_case_first_output(model_1, model_2, image, stride_xy, stride_z, patch_size, num_classes=1, alpha=0.5):
    w, h, d = image.shape

    # padding
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w; add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h; add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d; add_pad = True
    else:
        d_pad = 0

    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2

    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
                       mode='constant', constant_values=0)

    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])

                patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                patch = np.expand_dims(np.expand_dims(patch, axis=0), axis=0).astype(np.float32)
                patch = torch.from_numpy(patch).cuda()

                with torch.no_grad():
                    y1 = model_1(patch)
                    y2 = model_2(patch)

                    if isinstance(y1, (tuple, list)): y1 = y1[0]
                    if isinstance(y2, (tuple, list)): y2 = y2[0]

                    y1 = F.softmax(y1, dim=1)
                    y2 = F.softmax(y2, dim=1)

                    y = alpha * y1 + (1 - alpha) * y2

                y = y.cpu().data.numpy()[0]   # (C, D, H, W)

                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)

    if num_classes == 2:
        label_map = (score_map[1] > 0.5).astype(np.uint8)
    else:
        label_map = np.argmax(score_map, axis=0).astype(np.uint8)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]

    return label_map, score_map
  
def calculate_metric_percase(pred, gt):
    if pred.sum() == 0 or gt.sum() == 0:
        return 0, 0, 0, 0
    dice = metric.binary.dc(pred, gt)
    jc   = metric.binary.jc(pred, gt)
    hd   = metric.binary.hd95(pred, gt)
    asd  = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd

def calculate_metric_multiclass(pred, gt, num_classes):
    metrics = []
    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.uint8)
        gt_c   = (gt == c).astype(np.uint8)
        if pred_c.sum() == 0 or gt_c.sum() == 0:
            metrics.append([0, 0, 0, 0])
        else:
            metrics.append([
                metric.binary.dc(pred_c, gt_c),
                metric.binary.jc(pred_c, gt_c),
                metric.binary.hd95(pred_c, gt_c),
                metric.binary.asd(pred_c, gt_c),
            ])
    return np.array(metrics)