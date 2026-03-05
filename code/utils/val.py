import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import cv2
import os

def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    else:
        if pred.sum() == 0 and gt.sum() == 0:
            return 1.0
        else:
            return 0.0

def test_single_volume(image, label, model_1, model_2, classes, patch_size=[224, 224]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction_1 = np.zeros_like(label)
    prediction_2 = np.zeros_like(label)
    prediction_fusion = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model_1.eval()
        model_2.eval()
        with torch.no_grad():
            try:
                output_1, _ = model_1(input)
                output_2, _ = model_2(input)
            except:
                output_1 = model_1(input)
                output_2 = model_2(input)
            output_fusion = (output_1 + output_2) / 2
            out_1 = torch.argmax(torch.softmax(output_1, dim=1), dim=1).squeeze(0)
            out_1 = out_1.cpu().detach().numpy()
            pred_1 = zoom(out_1, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction_1[ind] = pred_1
            out_2 = torch.argmax(torch.softmax(output_2, dim=1), dim=1).squeeze(0)
            out_2 = out_2.cpu().detach().numpy()
            pred_2 = zoom(out_2, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction_2[ind] = pred_2
            out_fusion = torch.argmax(torch.softmax(output_fusion, dim=1), dim=1).squeeze(0)
            out_fusion = out_fusion.cpu().detach().numpy()
            pred_fusion = zoom(out_fusion, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction_fusion[ind] = pred_fusion
    metric_list_1 = []
    metric_list_2 = []
    metric_list_fusion = []
    for i in range(1, classes):
        metric_list_1.append(calculate_metric_percase(prediction_1 == i, label == i))
        metric_list_2.append(calculate_metric_percase(prediction_2 == i, label == i))
        metric_list_fusion.append(calculate_metric_percase(prediction_fusion == i, label == i))
    return metric_list_1, metric_list_2, metric_list_fusion