import os
import argparse
import torch

from networks.net_factory_3D import net_factory_3d
from utils.test_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,
                    default='BraTS2019', help='dataset_name')
parser.add_argument('--root_path', type=str, default='../',
                    help='Name of Experiment')
parser.add_argument('--exp', type=str, default='DCF', help='exp_name')
parser.add_argument('--model_1', type=str,
                    default='attention_unet', help='model_name')
parser.add_argument('--model_2', type=str,
                    default='voxresnet', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1,
                    help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=8, help='labeled data')
parser.add_argument('--label_ratio', type=str,
                    default='10%', help='label ratio')
parser.add_argument('--nms', type=int, default=0,
                    help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

label_num_mapping = {  
    "Pancreas_CT": {"1%": 2, '5%': 3, '10%': 6, '20%': 12, '30%': 19, '50%': 31, '70%': 43, '90%': 56, '100%': 62},  
    "BraTS2019": {"1%": 3, '5%': 12, '10%': 25, '20%': 50, '30%': 75, '50%': 125, '70%': 175, '90%': 225, '100%': 250}
}

FLAGS.labelnum = label_num_mapping.get(
    FLAGS.dataset_name, {}).get(FLAGS.label_ratio, None)

snapshot_path = "../model/{}_{}_{}_labeled".format(
    FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum)
test_save_path = "../model/{}_{}_{}_labeled/predictions/".format(
    FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum)

if FLAGS.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    FLAGS.root_path = FLAGS.root_path + 'data/Pancreas_CT/'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" +
                  item.replace('\n', '') + "_norm.h5" for item in image_list]
    num_classes = 2

elif FLAGS.dataset_name == "BraTS2019":
    patch_size = (96, 96, 96)
    FLAGS.root_path = FLAGS.root_path + 'data/BraTS2019/'
    with open(FLAGS.root_path + '/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/data/" +
                  item.replace('\n', '') + ".h5" for item in image_list]
    num_classes = 2
    
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)

def test_calculate_metric(FLAGS=FLAGS):

    net_1 = net_factory_3d(net_type=FLAGS.model_1,
                           in_chns=1, class_num=num_classes)
    net_2 = net_factory_3d(net_type=FLAGS.model_2,
                           in_chns=1, class_num=num_classes)

    save_mode_path_1 = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model_1))
    net_1.load_state_dict(torch.load(save_mode_path_1), strict=True)
    print("init weight from {}".format(save_mode_path_1))
    save_mode_path_2 = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model_2))
    net_2.load_state_dict(torch.load(save_mode_path_2), strict=True)
    print("init weight from {}".format(save_mode_path_2))

    net_1.eval()
    net_2.eval()

    if FLAGS.dataset_name == "Pancreas_CT":
        avg_metric = test_all_case(FLAGS.model_1,
                                   1,
                                   net_1,
                                   net_2,
                                   image_list,
                                   num_classes=num_classes,
                                   patch_size=(
                                       96, 96, 96),
                                   stride_xy=16,
                                   stride_z=16,
                                   save_result=True,
                                   test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail,
                                   nms=FLAGS.nms)

    elif FLAGS.dataset_name == "BraTS2019":
        avg_metric = test_all_case(FLAGS.model_1,
                                   1,
                                   net_1,
                                   net_2,
                                   image_list,
                                   num_classes=num_classes,
                                   patch_size=(
                                       96, 96, 96),
                                   stride_xy=64,
                                   stride_z=64,
                                   save_result=True,
                                   test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail,
                                   nms=FLAGS.nms)
    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
