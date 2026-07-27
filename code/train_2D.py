import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss,KLDivLoss,BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.dataset_2D import *
from networks.net_factory_2D import net_factory
from utils import losses, ramps, val, val_single
import math
import time

# torch.use_deterministic_algorithms(True)
# torch.autograd.set_detect_anomaly(True)

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='DCF', help='experiment_name')
parser.add_argument('--model_1', type=str, default='resunet_feature', help='model_name')
parser.add_argument('--model_2', type=str, default='swinunet_feature', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[224, 224], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--labeled_bs', type=int, default=6, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7, help='labeled data')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--label_ratio', type=str, default='10%', help='label ratio')
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
# self-config
parser.add_argument('--tau', type=float, default='0.7', help='pre_train model')
parser.add_argument('--base_temperature', type=float, default='0.1', help='pre_train model')
parser.add_argument('--gamma', type=float, default='0.5', help='pre_train model')
parser.add_argument('--lambda_c', type=float, default='0.1', help='pre_train model')


args = parser.parse_args()

label_num_mapping = {
    "ACDC": {'1%':1, '5%': 3, '10%': 7, '20%': 14, '30%': 21, '50%': 35, '70%': 49, '90%': 63, '100%': 70},
    "Prostate": {'1%':1, '5%': 2, '10%': 4, '20%': 7, '25%': 9, '30%': 11, '50%': 18, '70%': 25, '90%': 32, '100%': 35},
    "Hippocampus": {'1%': 2, '5%': 8, '10%': 16, '20%': 31, '30%': 47, '50%': 78, '70%': 109, '90%': 140, '100%': 156},
    "ATLAS":{'1%':1, '5%': 2, '10%': 4, '20%': 7, '30%': 11, '50%': 18, '70%': 25, '90%': 32, '100%': 36},
}

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if "ACDC" in args.root_path:
    args.labeled_num = label_num_mapping["ACDC"][args.label_ratio]
    args.num_classes = 4
    args.tau = 0.3
elif "Prostate" in args.root_path:
    args.labeled_num = label_num_mapping["Prostate"][args.label_ratio]
    args.num_classes = 2
    args.tau = 0.7
elif "Hippocampus" in args.root_path:
    args.labeled_num = label_num_mapping["Hippocampus"][args.label_ratio]
    args.num_classes = 3
    args.tau = 0.5
elif "ATLAS" in args.root_path:
    args.labeled_num = label_num_mapping["ATLAS"][args.label_ratio]
    args.num_classes = 3
    args.tau = 0.3
else:
    print("Error")
    quit()    
dice_loss = losses.DiceLoss(n_classes=args.num_classes)

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136, "14": 256, "21": 396, "35": 664, "49": 924, "63": 1178, "70": 1312}
    elif "Prostate" in dataset:
        ref_dict = {"1": 24, "2": 47, "4": 111, "7": 191, "9":258, '11':306, "18":478, "25":653, "32":835, "35": 940}
    elif "Hippocampus" in dataset:
        ref_dict = {"2": 73, "8": 288, "16": 581,"31": 1115, "47": 1688, "78": 2823, "109": 3905, "140": 4999, "156": 5559}
    elif "ATLAS" in dataset:
        ref_dict = {"1": 72, "2": 120, "4": 272, "7": 528, "11": 912, "18": 1372, "25": 1932, "32": 2536, "36": 2864}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*3/5), int(img_y*3/5)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y *4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce

def evidence_activation(logits):
    return F.softplus(logits)

def compute_belief_uncertainty(logits, num_classes):
    evidence = evidence_activation(logits)
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    belief = evidence / S
    uncertainty = num_classes / S
    
    return belief, uncertainty

def dempster_shafer_combination(b1, u1, b2, u2):    
    sum_b1 = 1.0 - u1
    sum_b2 = 1.0 - u2
    dot_prod = torch.sum(b1 * b2, dim=1, keepdim=True)
    
    conflict = (sum_b1 * sum_b2) - dot_prod
    conflict = torch.clamp(conflict, min=0.0, max=0.999)
    
    scale = 1.0 / (1.0 - conflict)
    
    b_fused = (b1 * b2 + b1 * u2 + u1 * b2) * scale
    u_fused = (u1 * u2) * scale
    
    return b_fused, u_fused, conflict

class UnifiedSemanticMemoryBank:  
    def __init__(self, num_classes, feature_dims=[256, 192], momentum=0.99):  
        super().__init__()  
        self.num_classes = num_classes  
        self.feature_dims = feature_dims  
        self.momentum = momentum  
        self.prototypes = [torch.zeros(num_classes, dim).cuda() for dim in feature_dims]  
        self.initialized = [False] * len(feature_dims)
        self.history_matrices = [torch.eye(num_classes).cuda() for _ in feature_dims]

    def update(self, features, labels, mask=None, gamma=0.05, stream_index=0):  
        with torch.no_grad():  
            if features.shape[2:] != labels.shape[1:]:  
                features = F.interpolate(features, size=labels.shape[1:], mode='bilinear', align_corners=True)  
            
            # Flatten features & labels
            features = features.permute(0, 2, 3, 1).reshape(-1, self.feature_dims[stream_index])  
            labels = labels.reshape(-1)  
            
            # Apply Masking (Convergence Guidance)
            if mask is not None:  
                mask = mask.reshape(-1)  
                valid_indices = mask > gamma
                features = features[valid_indices]  
                labels = labels[valid_indices]  
  
            if features.shape[0] == 0:  
                return  

            current_protos = self.prototypes[stream_index]
            
            for c in range(self.num_classes):  
                c_mask = (labels == c)  
                if c_mask.sum() > 0:  
                    c_feats = features[c_mask]  
                    
                    # 1. Update Prototype Center (Geometrical Mean)
                    c_mean = c_feats.mean(dim=0)  
                    c_mean = F.normalize(c_mean, dim=0)  
                    
                    if not self.initialized[stream_index]:  
                        self.prototypes[stream_index][c] = c_mean  
                    else:  
                        self.prototypes[stream_index][c] = (self.momentum * current_protos[c] + (1 - self.momentum) * c_mean)  
                        self.prototypes[stream_index][c] = F.normalize(self.prototypes[stream_index][c], dim=0)  
                    
                    with torch.no_grad():
                        sim_vector = torch.matmul(c_mean.unsqueeze(0), self.prototypes[stream_index].T)
                        sim_vector = (sim_vector + 1.0) / 2.0 # Normalize -1~1 to 0~1
                        if self.initialized[stream_index]:
                            self.history_matrices[stream_index][c] = (
                                0.999 * self.history_matrices[stream_index][c] + 
                                0.001 * sim_vector.squeeze(0)
                            )
                        else:
                            self.history_matrices[stream_index][c] = sim_vector.squeeze(0)

            self.initialized[stream_index] = True  
  
    def get_feature_driven_distribution(
                self, 
                features, 
                primary_prediction, 
                stream_index,
                tau = 0.7,
                base_temperature=0.1
            ):

            peer_index = 1 - stream_index
            
            if features.shape[2:] != primary_prediction.shape[1:]:
                features = F.interpolate(features, size=primary_prediction.shape[1:], mode='bilinear', align_corners=True)
                
            prototypes = self.prototypes[stream_index].detach()
            
            B, C, H, W = features.shape
            
            f_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
            f_norm = F.normalize(f_flat, dim=1)
            primary_flat = primary_prediction.reshape(-1)
            
            feature_prototype_sim = torch.matmul(f_norm, prototypes.T) 
            
            max_geo_sim, _ = torch.max(feature_prototype_sim, dim=1, keepdim=True)
            pure_level = (max_geo_sim + 1.0) / 2.0
            
            gamma = 10.0
            gate_alpha = torch.sigmoid(gamma * (pure_level - tau))

            my_history = self.history_matrices[stream_index][primary_flat]   
            peer_history = self.history_matrices[peer_index][primary_flat]   
            
            primary_protos_norm = F.normalize(prototypes[primary_flat], dim=1)
            geo_sim_vector = (torch.matmul(primary_protos_norm, prototypes.T) + 1.0) / 2.0

            # Deepening Unique Expertise
            dist_intro_base = (feature_prototype_sim + 1.0) / 2.0 
            unique_bias = my_history - peer_history 
            dist_intro = dist_intro_base + 0.3 * unique_bias
            temp_intro = base_temperature * 0.5
            
            # Orthogonal Space Expansion
            dissimilarity = 1.0 - geo_sim_vector 
            orthogonality = 1.0 - peer_history
            dist_extro = dissimilarity * orthogonality
            dist_extro = F.relu(dist_extro) + 0.01
            temp_extro = base_temperature * 2.0

            # Fusion: Construct Target Distribution Q
            logits_intro = dist_intro / temp_intro
            logits_extro = dist_extro / temp_extro
            
            final_logits_q = gate_alpha * logits_intro + (1.0 - gate_alpha) * logits_extro
            mask_self = F.one_hot(primary_flat, num_classes=self.num_classes).bool()
            final_logits_q.masked_fill_(mask_self, -1e9)
            probs_diff = F.softmax(final_logits_q, dim=1)
            
            return probs_diff.reshape(B, H, W, self.num_classes).permute(0, 3, 1, 2), gate_alpha.reshape(B, 1, H, W)

def pre_train(args, snapshot_path, model_name):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    if "swin" in model_name:
        model = net_factory(net_type=args.model_2, in_chns=1, class_num=num_classes, args=args).cuda()
    else:
        model = net_factory(net_type=args.model_1, in_chns=1, class_num=num_classes).cuda()

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([WeakStrongAugment(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labeled_num)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)

            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl, _ = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
                
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_single.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                performance = np.mean(metric_list, axis=0)[0]

                if performance >= best_performance:
                    best_performance = performance
                    if "swin" in model_name:
                        save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model_2))
                    else:
                        save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model_1))
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def train(args, snapshot_path, pre_snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    tau = args.tau
    base_temperature = args.base_temperature
    gamma = args.gamma

    model_1 = net_factory(net_type=args.model_1, in_chns=1, class_num=num_classes).cuda()
    save_model_path_1 = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model_1))
    model_1.load_state_dict(torch.load(save_model_path_1), strict=True)
    model_2 = net_factory(net_type=args.model_2, in_chns=1, class_num=num_classes, args=args).cuda()
    save_model_path_2 = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model_2))
    model_2.load_state_dict(torch.load(save_model_path_2), strict=True)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([
                                WeakStrongAugment(args.patch_size)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model_1.train()
    model_2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer_1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_2 = optim.Adam(model_2.parameters(),lr=base_lr*0.01,weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()

    mse_criterion = losses.mse_loss

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance_1 = 0.0
    best_performance_2 = 0.0
    best_performance_1_single = 0.0
    best_performance_2_single = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    memory_bank = None

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_weak_batch, volume_strong_batch = sampled_batch['image'].cuda(), sampled_batch["image_strong"].cuda()
            label_weak_batch, label_strong_batch = sampled_batch["label"].cuda(), sampled_batch["label_strong"].cuda()
            
            outputs_weak_1, feats_weak_1 = model_1(volume_weak_batch)
            outputs_strong_1, feats_strong_1 = model_1(volume_strong_batch)
            outputs_weak_2, feats_weak_2 = model_2(volume_weak_batch)
            outputs_strong_2, feats_strong_2 = model_2(volume_strong_batch)
            dim1 = feats_weak_1.shape[1]
            dim2 = feats_weak_2.shape[1]

            if memory_bank is None:
                memory_bank = UnifiedSemanticMemoryBank(num_classes=num_classes, feature_dims=[dim1, dim2])

            mse_loss_1 = mse_criterion(F.softmax(outputs_weak_1, dim=1), F.softmax(outputs_strong_1, dim=1))
            mse_loss_2 = mse_criterion(F.softmax(outputs_weak_2, dim=1), F.softmax(outputs_strong_2, dim=1))

            b_weak_1, u_weak_1 = compute_belief_uncertainty(outputs_weak_1, num_classes)
            b_strong_1, u_strong_1 = compute_belief_uncertainty(outputs_strong_1, num_classes)
            b_weak_2, u_weak_2 = compute_belief_uncertainty(outputs_weak_2, num_classes)
            b_strong_2, u_strong_2 = compute_belief_uncertainty(outputs_strong_2, num_classes)

            p_weak_1 = b_weak_1 + u_weak_1 / num_classes
            p_strong_1 = b_strong_1 + u_strong_1 / num_classes
            p_weak_2 = b_weak_2 + u_weak_2 / num_classes
            p_strong_2 = b_strong_2 + u_strong_2 / num_classes

            loss_seg_dice_1 = dice_loss(p_weak_1[:labeled_bs, ...], label_weak_batch[:labeled_bs].unsqueeze(1))
            loss_seg_dice_1 += dice_loss(p_strong_1[:labeled_bs, ...], label_strong_batch[:labeled_bs].unsqueeze(1))
            loss_seg_dice_2 = dice_loss(p_weak_2[:labeled_bs, ...], label_weak_batch[:labeled_bs].unsqueeze(1))
            loss_seg_dice_2 += dice_loss(p_strong_2[:labeled_bs, ...], label_strong_batch[:labeled_bs].unsqueeze(1))

            memory_bank.update(feats_weak_1[:labeled_bs], label_weak_batch[:labeled_bs], gamma=gamma, stream_index=0)
            memory_bank.update(feats_weak_2[:labeled_bs], label_weak_batch[:labeled_bs], gamma=gamma, stream_index=1)

            b_w1_unlab = b_weak_1[labeled_bs:]
            u_w1_unlab = u_weak_1[labeled_bs:]
            b_w2_unlab = b_weak_2[labeled_bs:]
            u_w2_unlab = u_weak_2[labeled_bs:]

            with torch.no_grad():
                b_fused, u_fused, conflict = dempster_shafer_combination(b_w1_unlab, u_w1_unlab, b_w2_unlab, u_w2_unlab)
                pseudo_label = torch.argmax(b_fused, dim=1)
                w_conv = (1.0 - conflict) * (1.0 - u_fused)
                w_explore = 1 - w_conv
                
            memory_bank.update(feats_weak_1[labeled_bs:], pseudo_label, w_conv, gamma, stream_index=0)
            memory_bank.update(feats_weak_2[labeled_bs:], pseudo_label, w_conv, gamma, stream_index=1)

            loss_self_map_1 = F.cross_entropy(outputs_weak_1[labeled_bs:], pseudo_label, reduction='none')
            loss_self_map_2 = F.cross_entropy(outputs_weak_2[labeled_bs:], pseudo_label, reduction='none')
            
            loss_self_1 = (w_conv * loss_self_map_1).mean()
            loss_self_2 = (w_conv * loss_self_map_2).mean()

            if all(memory_bank.initialized):
                q_counter_1, _ = memory_bank.get_feature_driven_distribution(
                    features=feats_weak_1[labeled_bs:], 
                    primary_prediction=pseudo_label, 
                    stream_index=0,
                    tau = tau,
                    base_temperature = base_temperature
                )

                q_counter_2, _ = memory_bank.get_feature_driven_distribution(
                    features=feats_weak_2[labeled_bs:], 
                    primary_prediction=pseudo_label, 
                    stream_index=1,
                    tau = tau,
                    base_temperature = base_temperature
                )
                
                log_p_1 = torch.log(p_weak_1[labeled_bs:] + 1e-6)
                log_p_2 = torch.log(p_weak_2[labeled_bs:] + 1e-6)
                
                loss_explore_1_map = -(q_counter_1 * log_p_1).sum(dim=1)
                loss_explore_2_map = -(q_counter_2 * log_p_2).sum(dim=1)

                loss_div_1 = (w_explore.squeeze(1) * loss_explore_1_map).mean()
                loss_div_2 = (w_explore.squeeze(1) * loss_explore_2_map).mean()
            else:
                loss_div_1 = torch.tensor(0.0).cuda()
                loss_div_2 = torch.tensor(0.0).cuda()

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            
            loss_1 = loss_seg_dice_1 + args.lambda_c * (loss_self_1 + loss_div_1) + consistency_weight * mse_loss_1
            loss_2 = loss_seg_dice_2 + args.lambda_c * (loss_self_2 + loss_div_2) + consistency_weight * mse_loss_2
            loss = loss_1 + loss_2

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            loss.backward()
            optimizer_1.step()
            optimizer_2.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer_1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer_2.param_groups:
                param_group['lr'] = lr_ * 0.01

            iter_num = iter_num + 1
            logging.info('model1 iteration %d : loss : %03f, supervised_loss: %03f, mse_loss: %03f, convergence_loss: %03f, divergence_loss: %03f' % (iter_num, loss_1, loss_seg_dice_1, mse_loss_1, loss_self_1, loss_div_1))
            logging.info('model2 iteration %d : loss : %03f, supervised_loss: %03f, mse_loss: %03f, convergence_loss: %03f, divergence_loss: %03f' % (iter_num, loss_2, loss_seg_dice_2, mse_loss_2, loss_self_2, loss_div_2))

            if iter_num > 0 and iter_num % 200 == 0:
                model_1.eval()
                model_2.eval()
                metric_list_1 = 0.0
                metric_list_2 = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i_1, metric_i_2 = val.test_single_volume(sampled_batch["image"], sampled_batch["label"], model_1, model_2, classes=num_classes)
                    metric_list_1 += np.array(metric_i_1)
                    metric_list_2 += np.array(metric_i_2)
                metric_list_1 = metric_list_1 / len(db_val)
                metric_list_2 = metric_list_2 / len(db_val)
                performance_1 = np.mean(metric_list_1, axis=0)
                performance_2 = np.mean(metric_list_2, axis=0)
                if performance_1 >= best_performance_1_single:
                    best_performance_1_single = performance_1
                    save_best_path_1_single = os.path.join(snapshot_path, '{}_best_model_single.pth'.format(args.model_1))
                    torch.save(model_1.state_dict(), save_best_path_1_single)
                if performance_2 >= best_performance_2_single:
                    best_performance_2_single = performance_2
                    save_best_path_2_single = os.path.join(snapshot_path, '{}_best_model_single.pth'.format(args.model_2))
                    torch.save(model_2.state_dict(), save_best_path_2_single)
                logging.info('model1 iteration %d : mean_dice : %f' % (iter_num, performance_1))
                logging.info('model2 iteration %d : mean_dice : %f' % (iter_num, performance_2))
                model_1.train()
                model_2.train()
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break
    writer.close()
    print(f"best_model_1: {best_performance_1:.4f}")
    print(f"best_model_2: {best_performance_2:.4f}")
    print("Training Finished!")

if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_{}_{}_{}_labeled".format(args.root_path.split("/")[-1], args.model_1, args.model_2, args.exp, args.labeled_num)
    pre_snapshot_path = "../model/{}_{}_{}_{}_{}_labeled/pre".format(args.root_path.split("/")[-1], args.model_1, args.model_2, args.exp, args.labeled_num)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copy('../code/train_2D.py', snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    torch.backends.cudnn.enabled = False

    pre_train(args, pre_snapshot_path, args.model_1)
    pre_train(args, pre_snapshot_path, args.model_2)

    train(args, snapshot_path, pre_snapshot_path)