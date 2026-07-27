import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils import ramps, losses, metrics, test_patch
from dataloaders.dataset_3D import *
from networks.net_factory_3D import net_factory_3d
from utils.BCP_utils import context_mask

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='BraTS2019', help='dataset_name')
parser.add_argument('--root_path', type=str, default='../', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='DCF', help='Saved folder name')
parser.add_argument('--model', type=str, default='voxresnet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--label_ratio', type=str, default='10%', help='label ratio')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')

args = parser.parse_args()

label_num_mapping = {  
    "Pancreas_CT": {"1": 1, '5%': 3, '10%': 6, '20%': 12, '30%': 19, '50%': 31, '70%': 43, '90%': 56, '100%': 62},  
    "BraTS2019": {"1": 3, '5%': 12, '10%': 25, '20%': 50, '30%': 75, '50%': 125, '70%': 175, '90%': 225, '100%': 250},
}

num_classes = 2
if args.dataset_name == "Pancreas_CT":  
    patch_size = (96, 96, 96)  
    args.root_path = args.root_path + 'data/Pancreas_CT/'  
    args.max_samples = 62  
    num_classes = 2
elif args.dataset_name == "BraTS2019":  
    patch_size = (96, 96, 96)  
    args.root_path = args.root_path + 'data/BraTS2019/'  
    args.max_samples = 250
    num_classes = 2

args.labelnum = label_num_mapping.get(args.dataset_name, {}).get(args.label_ratio, None)

snapshot_path = "../model/{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum, args.model)

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr
pre_max_iterations = args.pre_max_iteration

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(sys.argv[0])

    model = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes, shape=patch_size)
    if args.dataset_name == "Pancreas_CT":  
        db_train = Pancreas(base_dir=train_data_path,  
                            split='train',  
                            transform=transforms.Compose([  
                                RandomCrop(patch_size),  
                                ToTensor(),  
                            ]))  
    elif args.dataset_name == "BraTS2019":  
        db_train = BraTS2019(base_dir=train_data_path,  
                             split='train',  
                             transform=transforms.Compose([  
                                 RandomRotFlip(),  
                                 RandomCrop(patch_size),  
                                 ToTensor(),  
                             ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            try:
                outputs, _ = model(volume_batch)
            except:
                outputs = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f'%(iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                model.eval()
                if args.dataset_name == "Pancreas_CT":
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=16,
                                                          stride_z=16,
                                                          dataset_name='Pancreas_CT')
                elif args.dataset_name == "BraTS2019":
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=64,
                                                          stride_z=64,
                                                          dataset_name='BraTS2019')

                if dice_sample >= best_dice:
                    best_dice = dice_sample
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best_path)
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break
    print(best_dice)
