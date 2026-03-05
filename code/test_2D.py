import argparse
import os
import shutil
import logging
import sys
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
from networks.net_factory_2D import net_factory
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../../data/ATLAS', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='DCF', help='experiment_name')
parser.add_argument('--model_1', type=str, default='resunet_feature', help='model_name')
parser.add_argument('--model_2', type=str, default='swinunet_feature', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7, help='labeled data')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--label_ratio', type=str, default='10%', help='GPU to use')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--patch_size', type=list, default=[224, 224], help='patch size of network input')
# swinunet
parser.add_argument('--cfg', type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml",help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+', )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='no: no cache, ''full: cache all data, ''part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
args = parser.parse_args()

label_num_mapping = {
    "ACDC": {'5%': 3, '10%': 7, '15%':21, '20%': 14, '100%': 140},
    "Prostate": {'5%': 2, '10%': 4, '20%': 7, '100%': 35},
    "Hippocampus": {'5%': 8, '10%': 16, '20%': 31, '100%': 156},
    "ATLAS":{'5%': 2, '10%': 4, '20%': 7, '100%': 36},
}

if "ACDC" in args.root_path:
    args.labeled_num = label_num_mapping["ACDC"][args.label_ratio]
    args.num_classes = 4
elif "Prostate" in args.root_path:
    args.labeled_num = label_num_mapping["Prostate"][args.label_ratio]
    args.num_classes = 2
elif "Hippocampus" in args.root_path:
    args.labeled_num = label_num_mapping["Hippocampus"][args.label_ratio]
    args.num_classes = 3
elif "ATLAS" in args.root_path:
    args.labeled_num = label_num_mapping["ATLAS"][args.label_ratio]
    args.num_classes = 3
else:
    print("Error")
    sys.exit(1)

def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    if gt.sum() == 0:
        return None, None, None, None

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, jc, hd95, asd
    else:
        if pred.sum() == 0 and gt.sum() == 0:
            return 1.0, 1.0, 0.0, 0.0
        else:
            return 0.0, 0.0, None, None

def test_single_volume(case, model_1, model_2, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
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
            output = (output_1 + output_2) / 2
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
            prediction[ind] = pred

    per_class_metrics = []
    for cls in range(1, FLAGS.num_classes):
        if np.sum(label == cls) == 0:
            per_class_metrics.append((None, None, None, None))
        else:
            metrics = calculate_metric_percase(prediction == cls, label == cls)
            per_class_metrics.append(metrics)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))

    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")

    return per_class_metrics

def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "../model/{}_{}_{}_{}_{}_labeled".format(args.root_path[11:], args.model_1, args.model_2, args.exp, args.labeled_num)
    test_save_path = "../model/{}_{}_{}_{}_{}_labeled/predictions/".format(args.root_path[11:], args.model_1, args.model_2, args.exp, args.labeled_num)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    logging.basicConfig(filename=snapshot_path + "/detail.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    model_1 = net_factory(net_type=FLAGS.model_1, in_chns=1, class_num=FLAGS.num_classes, args=FLAGS)
    model_2 = net_factory(net_type=FLAGS.model_2, in_chns=1, class_num=FLAGS.num_classes, args=FLAGS)
    save_model_path_1 = os.path.join(snapshot_path, '{}_best_model_single.pth'.format(FLAGS.model_1))
    save_model_path_2 = os.path.join(snapshot_path, '{}_best_model_single.pth'.format(FLAGS.model_2))
    model_1.load_state_dict(torch.load(save_model_path_1), strict=False)
    model_2.load_state_dict(torch.load(save_model_path_2), strict=False)
    model_1 = model_1.cuda()
    model_2 = model_2.cuda()
    print("init weight from {}".format(save_model_path_1))
    print("init weight from {}".format(save_model_path_2))
    model_1.eval()
    model_2.eval()

    total_metric_per_class = np.zeros((FLAGS.num_classes - 1, 4), dtype=np.float64)
    all_metrics = []

    for case in tqdm(image_list):
        case_metrics = test_single_volume(case, model_1, model_2, test_save_path, FLAGS)
        case_metrics = np.asarray(case_metrics, dtype=np.float64)

        total_metric_per_class += case_metrics

        print(f"\n{case} results:")
        row_metrics = [case]

        for cls in range(1, FLAGS.num_classes):
            dice, jc, hd95, asd = case_metrics[cls - 1]
            print(f"Model Class {cls:2d}: Dice: {dice:.6f}, JC: {jc:.6f}, HD95: {hd95:.6f}, ASD: {asd:.6f}")
            row_metrics.extend([dice, jc, hd95, asd])

        all_metrics.append(row_metrics)

    avg_metric_per_class = total_metric_per_class / len(image_list)

    header = ['Sample']
    for cls in range(1, FLAGS.num_classes):
        header.extend([f'Class {cls} Dice', f'Class {cls} JC', f'Class {cls} HD95', f'Class {cls} ASD'])
    
    csv_file = os.path.join(test_save_path, 'metrics_per_sample.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_metrics)

    return avg_metric_per_class, test_save_path


if __name__ == '__main__':
    metric, test_save_path = Inference(args)

    for cls in range(1, args.num_classes):
        print(f"Model Class {cls:2d} => Dice: {metric[cls - 1, 0]:.6f}, JC: {metric[cls - 1, 1]:.6f}, HD95: {metric[cls - 1, 2]:.6f}, ASD: {metric[cls - 1, 3]:.6f}")

    mean_avg = np.nanmean(metric, axis=0)
    print(f"\nModel Overall Average => Dice: {mean_avg[0]:.6f}, JC: {mean_avg[1]:.6f}, HD95: {mean_avg[2]:.6f}, ASD: {mean_avg[3]:.6f}")

    with open(test_save_path + '../performance.txt', 'w') as f:
        f.writelines("Per-class metrics:\n")
        for cls in range(1, args.num_classes):
            f.writelines(f"Model Class {cls:2d} => Dice: {metric[cls - 1, 0]:.6f}, JC: {metric[cls - 1, 1]:.6f}, HD95: {metric[cls - 1, 2]:.6f}, ASD: {metric[cls - 1, 3]:.6f}\n")
        f.writelines(f"\nModel Overall Average => Dice: {mean_avg[0]:.6f}, JC: {mean_avg[1]:.6f}, HD95: {mean_avg[2]:.6f}, ASD: {mean_avg[3]:.6f}\n")
