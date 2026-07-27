import os  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  
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
  
def evidence_activation(logits):  
    return F.softplus(logits)  
  
def resize_prob_map(prob, target_size):  
    if prob.shape[2:] != target_size:  
        prob = F.interpolate(  
            prob,  
            size=target_size,  
            mode='trilinear',  
            align_corners=False  
        )  
  
        prob = torch.clamp(prob, min=1e-7, max=1.0)
        prob = prob / (prob.sum(dim=1, keepdim=True) + 1e-7)
        
        prob = torch.where(
            torch.isnan(prob) | torch.isinf(prob),
            torch.ones_like(prob) / prob.shape[1],
            prob
        )
  
    return prob  
  
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

def spatial_consistency_refinement(pseudo_label, confidence, num_classes, kernel_size=5):
    B, D, H, W = pseudo_label.shape

    sigma = kernel_size / 6.0
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size).to(pseudo_label.device)

    pseudo_onehot = F.one_hot(pseudo_label, num_classes=num_classes)\
                     .permute(0, 4, 1, 2, 3).float()
    confidence_expanded = confidence.unsqueeze(1).expand_as(pseudo_onehot)
    weighted_onehot = pseudo_onehot * confidence_expanded

    smoothed = []
    for c in range(num_classes):
        smoothed_c = F.conv3d(
            weighted_onehot[:, c:c+1], kernel, padding=kernel_size // 2
        )
        smoothed.append(smoothed_c)
    smoothed = torch.cat(smoothed, dim=1)

    refined_pseudo = torch.argmax(smoothed, dim=1)
    refined_conf = smoothed.max(dim=1)[0]

    high_conf_mask = confidence > 0.8
    refined_pseudo = torch.where(high_conf_mask, pseudo_label, refined_pseudo)
    refined_conf = torch.where(high_conf_mask, confidence, refined_conf)

    return refined_pseudo, refined_conf

class UnifiedSemanticMemoryBank:  
    def __init__(  
            self,  
            num_classes,  
            feature_dims=[256, 256],  
            momentum=0.95,
            max_voxels_per_class=16384,
            label_smoothing=0.05,  
            self_suppress_binary=0.05,  
            self_suppress_multi=0.5,
            ema_momentum=0.999
    ):  
        super().__init__()  
  
        self.num_classes = num_classes  
        self.feature_dims = feature_dims  
        self.momentum = momentum  
        self.max_voxels_per_class = max_voxels_per_class  
        self.label_smoothing = label_smoothing  
        self.self_suppress_binary = self_suppress_binary  
        self.self_suppress_multi = self_suppress_multi  
        self.ema_momentum = ema_momentum
  
        self.prototypes = [  
            torch.zeros(num_classes, dim).cuda()  
            for dim in feature_dims  
        ]  
  
        self.initialized = [False] * len(feature_dims)  
  
        self.class_initialized = [  
            torch.zeros(num_classes, dtype=torch.bool).cuda()  
            for _ in feature_dims  
        ]  
  
        self.history_matrices = [  
            torch.eye(num_classes).cuda()  
            for _ in feature_dims  
        ]
        
        self.feature_ema = [None, None]
  
    def _prepare_labels_to_feature_size(self, labels, feature_spatial_size):  
        if labels.dim() == 5:  
            labels = labels.squeeze(1)  
  
        labels = labels.long()  
  
        if labels.shape[1:] != feature_spatial_size:  
            labels = F.interpolate(  
                labels.unsqueeze(1).float(),  
                size=feature_spatial_size,  
                mode='nearest'  
            ).squeeze(1).long()  
  
        return labels  
  
    def _prepare_mask_to_feature_size(self, mask, feature_spatial_size):  
        if mask is None:  
            return None  
  
        if mask.dim() == 4:  
            mask = mask.unsqueeze(1)  
  
        mask = mask.float()  
  
        if mask.shape[2:] != feature_spatial_size:  
            mask = F.interpolate(  
                mask,  
                size=feature_spatial_size,  
                mode='trilinear',  
                align_corners=False  
            )  
  
        return mask  

    def update_with_clustering(
            self,
            features,
            labels,
            mask=None,
            gamma=0.05,
            stream_index=0,
            use_pseudo=False,
            max_voxels_per_class=None
    ):
        with torch.no_grad():
            if max_voxels_per_class is None:
                max_voxels_per_class = self.max_voxels_per_class

            device = features.device
            self.prototypes[stream_index] = self.prototypes[stream_index].to(device)
            self.history_matrices[stream_index] = self.history_matrices[stream_index].to(device)
            self.class_initialized[stream_index] = self.class_initialized[stream_index].to(device)

            B, C, d, h, w = features.shape
            feature_spatial_size = features.shape[2:]

            if self.feature_ema[stream_index] is None:
                self.feature_ema[stream_index] = features.detach().clone()
            else:
                self.feature_ema[stream_index] = (
                    self.ema_momentum * self.feature_ema[stream_index] +
                    (1 - self.ema_momentum) * features.detach()
                )

            ema_feats = self.feature_ema[stream_index]
            ema_flat = ema_feats.permute(0, 2, 3, 4, 1).reshape(-1, C)
            ema_norm = F.normalize(ema_flat, dim=1)

            if not use_pseudo:
                labels_lr = self._prepare_labels_to_feature_size(
                    labels,
                    feature_spatial_size
                )
                labels_flat = labels_lr.reshape(-1).long()

                if mask is not None:
                    mask_lr = self._prepare_mask_to_feature_size(mask, feature_spatial_size)
                    mask_flat = mask_lr.reshape(-1)
                    valid_indices = mask_flat > gamma

                    ema_norm = ema_norm[valid_indices]
                    labels_flat = labels_flat[valid_indices]

                if ema_norm.shape[0] == 0:
                    return

                for c in range(self.num_classes):
                    c_indices = torch.where(labels_flat == c)[0]

                    if c_indices.numel() == 0:
                        continue

                    if c_indices.numel() > max_voxels_per_class:
                        perm = torch.randperm(c_indices.numel(), device=device)
                        c_indices = c_indices[perm[:max_voxels_per_class]]

                    c_feats = ema_norm[c_indices]

                    if c_feats.shape[0] == 0:
                        continue

                    c_mean = c_feats.mean(dim=0)
                    c_mean = F.normalize(c_mean, dim=0)

                    if not self.class_initialized[stream_index][c]:
                        self.prototypes[stream_index][c] = c_mean
                        self.class_initialized[stream_index][c] = True
                    else:
                        old_proto = self.prototypes[stream_index][c]
                        new_proto = self.momentum * old_proto + (1.0 - self.momentum) * c_mean
                        self.prototypes[stream_index][c] = F.normalize(new_proto, dim=0)

                    proto_norm = F.normalize(self.prototypes[stream_index], dim=1)
                    sim_vector = torch.matmul(
                        c_mean.unsqueeze(0),
                        proto_norm.T
                    )
                    sim_vector = (sim_vector + 1.0) / 2.0

                    if self.initialized[stream_index]:
                        self.history_matrices[stream_index][c] = (
                            0.999 * self.history_matrices[stream_index][c]
                            + 0.001 * sim_vector.squeeze(0)
                        )
                    else:
                        self.history_matrices[stream_index][c] = sim_vector.squeeze(0)

            else:
                if not self.initialized[stream_index]:
                    return

                prototypes_norm = F.normalize(
                    self.prototypes[stream_index].detach(), dim=1
                )

                sim_matrix = torch.matmul(ema_norm, prototypes_norm.T)

                temperature = 0.1
                soft_assignment = F.softmax(sim_matrix / temperature, dim=1)

                max_conf, _ = soft_assignment.max(dim=1)
                reliable_mask = max_conf > 0.7

                if reliable_mask.sum() == 0:
                    return

                ema_norm_reliable = ema_norm[reliable_mask]
                soft_assignment_reliable = soft_assignment[reliable_mask]

                for c in range(self.num_classes):
                    weights = soft_assignment_reliable[:, c:c+1]  # [N, 1]

                    if weights.shape[0] > max_voxels_per_class:
                        probs = weights.squeeze(1)
                        probs = probs / (probs.sum() + 1e-6)
                        sampled_indices = torch.multinomial(
                            probs,
                            max_voxels_per_class,
                            replacement=False
                        )
                        weights = weights[sampled_indices]
                        ema_norm_sampled = ema_norm_reliable[sampled_indices]
                    else:
                        ema_norm_sampled = ema_norm_reliable

                    weighted_feats = ema_norm_sampled * weights
                    c_mean = weighted_feats.sum(dim=0) / (weights.sum() + 1e-6)
                    c_mean = F.normalize(c_mean, dim=0)

                    if self.class_initialized[stream_index][c]:
                        old_proto = self.prototypes[stream_index][c]
                        unsup_momentum = 0.98
                        new_proto = unsup_momentum * old_proto + (1.0 - unsup_momentum) * c_mean
                        self.prototypes[stream_index][c] = F.normalize(new_proto, dim=0)

            self.initialized[stream_index] = bool(
                self.class_initialized[stream_index].all().item()
            )

    def update(
            self,
            features,
            labels,
            mask=None,
            gamma=0.05,
            stream_index=0,
            max_voxels_per_class=None
    ):
        self.update_with_clustering(
            features=features,
            labels=labels,
            mask=mask,
            gamma=gamma,
            stream_index=stream_index,
            use_pseudo=False,
            max_voxels_per_class=max_voxels_per_class
        )
  
    def get_feature_driven_distribution(  
            self,  
            features,  
            primary_prediction,  
            stream_index,  
            tau=0.7,  
            base_temperature=0.15
    ):  
        device = features.device  
  
        peer_index = 1 - stream_index  
  
        self.prototypes[stream_index] = self.prototypes[stream_index].to(device)  
        self.prototypes[peer_index] = self.prototypes[peer_index].to(device)  
        self.history_matrices[stream_index] = self.history_matrices[stream_index].to(device)  
        self.history_matrices[peer_index] = self.history_matrices[peer_index].to(device)  
  
        B, C, d, h, w = features.shape  
        feature_spatial_size = features.shape[2:]  
  
        primary_lr = self._prepare_labels_to_feature_size(  
            primary_prediction,  
            feature_spatial_size  
        )  
  
        f_flat = features.permute(0, 2, 3, 4, 1).reshape(-1, C)  
        f_norm = F.normalize(f_flat, dim=1)  
  
        primary_flat = primary_lr.reshape(-1).long()  
        primary_flat = torch.clamp(primary_flat, min=0, max=self.num_classes - 1)  
  
        prototypes = self.prototypes[stream_index].detach()  
        prototypes_norm = F.normalize(prototypes, dim=1)  
  
        feature_prototype_sim = torch.matmul(f_norm, prototypes_norm.T)  
  
        max_geo_sim, _ = torch.max(feature_prototype_sim, dim=1, keepdim=True)  
        pure_level = (max_geo_sim + 1.0) / 2.0  
  
        gamma_gate = 10.0  
        gate_alpha = torch.sigmoid(gamma_gate * (pure_level - tau))  
  
        my_history = self.history_matrices[stream_index][primary_flat]  
        peer_history = self.history_matrices[peer_index][primary_flat]  
  
        primary_protos_norm = prototypes_norm[primary_flat]  
  
        geo_sim_vector = torch.matmul(  
            primary_protos_norm,  
            prototypes_norm.T  
        )  
  
        geo_sim_vector = (geo_sim_vector + 1.0) / 2.0  
  
        dist_intro_base = (feature_prototype_sim + 1.0) / 2.0  
        unique_bias = my_history - peer_history  
        dist_intro = dist_intro_base + 0.3 * unique_bias  
  
        dist_intro = torch.clamp(dist_intro, min=1e-6)  
  
        temp_intro = base_temperature * 1.0
  
        dissimilarity = 1.0 - geo_sim_vector  
        orthogonality = 1.0 - peer_history  
        dist_extro = dissimilarity * orthogonality  
        dist_extro = F.relu(dist_extro) + 0.01  
  
        temp_extro = base_temperature * 3.0
  
        logits_intro = dist_intro / temp_intro  
        logits_extro = dist_extro / temp_extro  
  
        final_logits_q = gate_alpha * logits_intro + (1.0 - gate_alpha) * logits_extro  
  
        if self.num_classes == 2:  
            self_suppress = self.self_suppress_binary  
        else:  
            self_suppress = self.self_suppress_multi  
  
        mask_self = F.one_hot(  
            primary_flat,  
            num_classes=self.num_classes  
        ).bool()  
  
        final_logits_q = torch.where(  
            mask_self,  
            final_logits_q - self_suppress,  
            final_logits_q  
        )  
  
        probs_diff = F.softmax(final_logits_q, dim=1)  
  
        eps = self.label_smoothing  
        if eps > 0:  
            probs_diff = (1.0 - eps) * probs_diff + eps / self.num_classes  
            probs_diff = probs_diff / (probs_diff.sum(dim=1, keepdim=True) + 1e-6)  
  
        probs_diff = probs_diff.reshape(  
            B, d, h, w, self.num_classes  
        ).permute(0, 4, 1, 2, 3).contiguous()  
  
        gate_alpha = gate_alpha.reshape(  
            B, 1, d, h, w  
        ).contiguous()  
  
        return probs_diff, gate_alpha  
      
def get_current_consistency_weight(epoch):  
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)  
  
  
parser = argparse.ArgumentParser()  
parser.add_argument('--dataset_name', type=str, default='BraTS2019', help='dataset_name')  
parser.add_argument('--root_path', type=str, default='../', help='Name of Dataset')  
parser.add_argument('--exp', type=str, default='DCF', help='Saved folder name')  
parser.add_argument('--model_1', type=str, default='attention_unet', help='model_name')  
parser.add_argument('--model_2', type=str, default='voxresnet', help='model_name')  
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='maximum pre-train iteration to train')  
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
parser.add_argument('--consistency_rampup', type=float, default=150.0, help='consistency_rampup')  
parser.add_argument('--tau', type=float, default=0.7, help='pre_train model')  
parser.add_argument('--base_temperature', type=float, default=0.15, help='pre_train model') 
parser.add_argument('--gamma', type=float, default=0.7, help='pre_train model')  
parser.add_argument('--lambda_c', type=float, default=0.1, help='pre_train model')  
parser.add_argument('--stage1_iters', type=int, default=3000, help='stage 1 iterations (supervised only)')
parser.add_argument('--stage2_iters', type=int, default=8000, help='stage 2 iterations (gradual unsupervised)')
parser.add_argument('--use_spatial_refine', type=int, default=1, help='use spatial consistency refinement')
parser.add_argument('--use_augmentation', type=int, default=1, help='use different augmentation for two models')
args = parser.parse_args()
    
label_num_mapping = {  
    "Pancreas_CT": {"1%": 2, '5%': 3, '10%': 6, '20%': 12, '30%': 19, '50%': 31, '70%': 43, '90%': 56, '100%': 62},  
    "BraTS2019": {"1%": 3, '5%': 12, '10%': 25, '20%': 50, '30%': 75, '50%': 125, '70%': 175, '90%': 225, '100%': 250}
}
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
  
snapshot_path = "../model/{}_{}_{}_labeled".format(args.dataset_name, args.exp, args.labelnum)  
  
train_data_path = args.root_path  
  
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  
labeled_bs = args.labeled_bs  
max_iterations = args.max_iteration  
base_lr = args.base_lr  
base_temperature = args.base_temperature  
tau = args.tau  
gamma = args.gamma  
lambda_c = args.lambda_c  
  
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
  
    model_1 = net_factory_3d(net_type=args.model_1, in_chns=1, class_num=num_classes)  
    save_model_path_1 = os.path.join(snapshot_path, args.model_1,'{}_best_model.pth'.format(args.model_1))
    model_1.load_state_dict(torch.load(save_model_path_1), strict=True)  
    model_2 = net_factory_3d(net_type=args.model_2, in_chns=1, class_num=num_classes)  
    save_model_path_2 = os.path.join(snapshot_path, args.model_2,'{}_best_model.pth'.format(args.model_2))
    model_2.load_state_dict(torch.load(save_model_path_2), strict=True)
  
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
  
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)  
  
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train,  
                             batch_sampler=batch_sampler,  
                             num_workers=4,  
                             pin_memory=True,  
                             worker_init_fn=worker_init_fn)  
  
    optimizer_1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)  
    optimizer_2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)  
  
    writer = SummaryWriter(snapshot_path + '/log')  
    logging.info("{} itertations per epoch".format(len(trainloader)))  
    consistency_criterion = nn.CrossEntropyLoss(reduction='none')  
    dice_loss = losses.mask_DiceLoss(nclass=num_classes)  
  
    iter_num = 0  
    best_dice_1 = 0  
    best_dice_2 = 0  
    max_epoch = max_iterations // len(trainloader) + 1  
    lr_ = base_lr  
    iterator = tqdm(range(max_epoch), ncols=70)  
    memory_bank = None  
  
    for epoch_num in iterator:  
        for i_batch, sampled_batch in enumerate(trainloader):  
            scaler = torch.cuda.amp.GradScaler()  
  
            with torch.cuda.amp.autocast():  
  
                volume_weak_batch = sampled_batch['image'].cuda()  
                label_weak_batch = sampled_batch["label"].cuda()  
  
                # ✅ 新增：为两个模型生成不同的augmented view（打破共识）
                if args.use_augmentation and iter_num >= args.stage1_iters:
                    volume_unlab = volume_weak_batch[labeled_bs:]
                    
                    # Model 1: 添加高斯噪声
                    with torch.no_grad():
                        noise_1 = torch.randn_like(volume_unlab) * 0.1
                        volume_aug_1 = torch.cat([
                            volume_weak_batch[:labeled_bs],
                            volume_unlab + noise_1
                        ], dim=0)
                    
                    # Model 2: 添加dropout噪声
                    with torch.no_grad():
                        mask_2 = (torch.rand_like(volume_unlab) > 0.1).float()
                        volume_aug_2 = torch.cat([
                            volume_weak_batch[:labeled_bs],
                            volume_unlab * mask_2
                        ], dim=0)
                    
                    outputs_weak_1, feats_weak_1 = model_1(volume_aug_1)
                    outputs_weak_2, feats_weak_2 = model_2(volume_aug_2)
                else:
                    outputs_weak_1, feats_weak_1 = model_1(volume_weak_batch)  
                    outputs_weak_2, feats_weak_2 = model_2(volume_weak_batch)
  
                dim1 = feats_weak_1.shape[1]  
                dim2 = feats_weak_2.shape[1]  
  
                if memory_bank is None:  
                    memory_bank = UnifiedSemanticMemoryBank(  
                        num_classes=num_classes,  
                        feature_dims=[dim1, dim2],  
                        momentum=0.95,  
                        max_voxels_per_class=16384,  
                        label_smoothing=0.05,  
                        self_suppress_binary=0.05,  
                        self_suppress_multi=0.5,
                        ema_momentum=0.999
                    )  
  
                b_weak_1, u_weak_1 = compute_belief_uncertainty(outputs_weak_1, num_classes)  
                b_weak_2, u_weak_2 = compute_belief_uncertainty(outputs_weak_2, num_classes)  
  
                p_weak_1 = b_weak_1 + u_weak_1 / num_classes  
                p_weak_2 = b_weak_2 + u_weak_2 / num_classes  
  
                # 有监督损失
                loss_seg_dice_1 = dice_loss(p_weak_1[:labeled_bs, ...], label_weak_batch[:labeled_bs].unsqueeze(1))  
                loss_seg_dice_2 = dice_loss(p_weak_2[:labeled_bs, ...], label_weak_batch[:labeled_bs].unsqueeze(1))  
  
                loss_sup_ce_1 = F.nll_loss(  
                    torch.log(p_weak_1[:labeled_bs] + 1e-6),  
                    label_weak_batch[:labeled_bs].long()  
                )  
  
                loss_sup_ce_2 = F.nll_loss(  
                    torch.log(p_weak_2[:labeled_bs] + 1e-6),  
                    label_weak_batch[:labeled_bs].long()  
                )  
  
                # ✅ 核心修改：分阶段训练策略
                if iter_num < args.stage1_iters:
                    # ===== 阶段1：只用有监督损失，用GT更新memory bank =====
                    memory_bank.update_with_clustering(
                        feats_weak_1[:labeled_bs],
                        label_weak_batch[:labeled_bs],
                        gamma=gamma,
                        stream_index=0,
                        use_pseudo=False
                    )
                    memory_bank.update_with_clustering(
                        feats_weak_2[:labeled_bs],
                        label_weak_batch[:labeled_bs],
                        gamma=gamma,
                        stream_index=1,
                        use_pseudo=False
                    )
                    
                    loss_1 = loss_seg_dice_1 + 0.5 * loss_sup_ce_1
                    loss_2 = loss_seg_dice_2 + 0.5 * loss_sup_ce_2
                    
                    loss_self_1 = torch.tensor(0.0).cuda()
                    loss_self_2 = torch.tensor(0.0).cuda()
                    loss_div_1 = torch.tensor(0.0).cuda()
                    loss_div_2 = torch.tensor(0.0).cuda()
                    
                    logging.info(
                        'Stage 1 (Supervised Only) - iteration %d : '
                        'loss_1: %.3f, loss_2: %.3f, '
                        'dice_1: %.3f, dice_2: %.3f' %
                        (iter_num, loss_1.item(), loss_2.item(),
                         loss_seg_dice_1.item(), loss_seg_dice_2.item())
                    )
                
                else:
                    memory_bank.update_with_clustering(
                        feats_weak_1[:labeled_bs],
                        label_weak_batch[:labeled_bs],
                        gamma=gamma,
                        stream_index=0,
                        use_pseudo=False
                    )
                    memory_bank.update_with_clustering(
                        feats_weak_2[:labeled_bs],
                        label_weak_batch[:labeled_bs],
                        gamma=gamma,
                        stream_index=1,
                        use_pseudo=False
                    )
                    
                    b_w1_unlab = b_weak_1[labeled_bs:]  
                    u_w1_unlab = u_weak_1[labeled_bs:]  
                    b_w2_unlab = b_weak_2[labeled_bs:]  
                    u_w2_unlab = u_weak_2[labeled_bs:]  
      
                    with torch.no_grad():  
                        b_fused, u_fused, conflict = dempster_shafer_combination(  
                            b_w1_unlab, u_w1_unlab, b_w2_unlab, u_w2_unlab  
                        )  
      
                        pseudo_label_raw = torch.argmax(b_fused, dim=1)  
                        w_conv_raw = (1.0 - conflict) * (1.0 - u_fused)  
                        w_conv_raw = w_conv_raw.clamp(0.0, 1.0)
                        
                        if args.use_spatial_refine:
                            pseudo_label, w_conv_refined = spatial_consistency_refinement(
                                pseudo_label_raw,
                                w_conv_raw.squeeze(1),
                                num_classes=num_classes,
                                kernel_size=5
                            )
                            w_conv = w_conv_refined.unsqueeze(1)
                        else:
                            pseudo_label = pseudo_label_raw
                            w_conv = w_conv_raw
                        
                        w_explore = 2.0 * w_conv * (1.0 - w_conv)
                        w_explore = w_explore.clamp(0.0, 0.5).detach()
                    
                    memory_bank.update_with_clustering(
                        feats_weak_1[labeled_bs:],
                        None,
                        gamma=0.5,
                        stream_index=0,
                        use_pseudo=True
                    )
                    memory_bank.update_with_clustering(
                        feats_weak_2[labeled_bs:],
                        None,
                        gamma=0.5,
                        stream_index=1,
                        use_pseudo=True
                    )
                    
                    log_p_1_unlab = torch.log(p_weak_1[labeled_bs:] + 1e-6)  
                    log_p_2_unlab = torch.log(p_weak_2[labeled_bs:] + 1e-6)  
      
                    loss_self_map_1 = F.nll_loss(log_p_1_unlab, pseudo_label, reduction='none')  
                    loss_self_map_2 = F.nll_loss(log_p_2_unlab, pseudo_label, reduction='none')  
      
                    fg_ratio = (pseudo_label.float().mean() + 1e-6)
                    fg_weight = torch.clamp(1.0 / (fg_ratio + 0.1), min=1.5, max=2.5)
                    
                    pseudo_weight = torch.ones_like(loss_self_map_1)  
                    pseudo_weight[pseudo_label == 1] = fg_weight  
      
                    w_self = w_conv.squeeze(1) * pseudo_weight  
      
                    loss_self_1 = (w_self * loss_self_map_1).sum() / (w_self.sum() + 1e-6)  
                    loss_self_2 = (w_self * loss_self_map_2).sum() / (w_self.sum() + 1e-6)
                    
                    if all(memory_bank.initialized):  
                        with torch.no_grad():  
                            q_counter_1, _ = memory_bank.get_feature_driven_distribution(  
                                features=feats_weak_1[labeled_bs:],  
                                primary_prediction=pseudo_label,  
                                stream_index=0,  
                                tau=tau,  
                                base_temperature=base_temperature  
                            )  
      
                            q_counter_2, _ = memory_bank.get_feature_driven_distribution(  
                                features=feats_weak_2[labeled_bs:],  
                                primary_prediction=pseudo_label,  
                                stream_index=1,  
                                tau=tau,  
                                base_temperature=base_temperature  
                            )  
      
                            target_size_1 = p_weak_1[labeled_bs:].shape[2:]  
                            target_size_2 = p_weak_2[labeled_bs:].shape[2:]  
      
                            q_counter_1 = resize_prob_map(q_counter_1, target_size_1)  
                            q_counter_2 = resize_prob_map(q_counter_2, target_size_2)  
      
                        log_p_1 = torch.log(p_weak_1[labeled_bs:] + 1e-6)  
                        log_p_2 = torch.log(p_weak_2[labeled_bs:] + 1e-6)  
      
                        loss_explore_1_map = -(q_counter_1 * log_p_1).sum(dim=1)  
                        loss_explore_2_map = -(q_counter_2 * log_p_2).sum(dim=1)  
      
                        explore_mask_sum = w_explore.sum() + 1e-6  
      
                        loss_div_1 = (  
                            w_explore.squeeze(1) * loss_explore_1_map  
                        ).sum() / explore_mask_sum  
      
                        loss_div_2 = (  
                            w_explore.squeeze(1) * loss_explore_2_map  
                        ).sum() / explore_mask_sum  
      
                    else:  
                        loss_div_1 = torch.tensor(0.0).cuda()  
                        loss_div_2 = torch.tensor(0.0).cuda()
                    
                    if iter_num < args.stage2_iters:
                        progress = (iter_num - args.stage1_iters) / (args.stage2_iters - args.stage1_iters)
                        consistency_weight = args.lambda_c * progress
                        lambda_div = 0.0
                    else:
                        consistency_weight = get_current_consistency_weight(iter_num // 150)
                        consistency_weight = args.lambda_c * consistency_weight
                        lambda_div = 0.3 if all(memory_bank.initialized) else 0.0
                    
                    loss_1 = loss_seg_dice_1 + 0.5 * loss_sup_ce_1 + consistency_weight * (loss_self_1 + lambda_div * loss_div_1)  
                    loss_2 = loss_seg_dice_2 + 0.5 * loss_sup_ce_2 + consistency_weight * (loss_self_2 + lambda_div * loss_div_2)
                    
                    stage_name = "Stage 2 (Gradual)" if iter_num < args.stage2_iters else "Stage 3 (Full)"
                    logging.info(  
                        '%s - model1 iteration %d : loss: %.3f, supervised_loss: %.3f, convergence_loss: %.3f, divergence_loss: %.3f' %  
                        (stage_name, iter_num, loss_1.item(), loss_seg_dice_1.item(), loss_self_1.item(), loss_div_1.item()))  
      
                    logging.info(  
                        '%s - model2 iteration %d : loss: %.3f, supervised_loss: %.3f, convergence_loss: %.3f, divergence_loss: %.3f' %  
                        (stage_name, iter_num, loss_2.item(), loss_seg_dice_2.item(), loss_self_2.item(), loss_div_2.item()))  
                      
                loss = loss_1 + loss_2
  
            optimizer_1.zero_grad()  
            optimizer_2.zero_grad()  
  
            scaler.scale(loss).backward()  
            scaler.step(optimizer_1)  
            scaler.step(optimizer_2)  
            scaler.update()  
  
            iter_num = iter_num + 1  
  
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9  
            for param_group in optimizer_1.param_groups:  
                param_group['lr'] = lr_  
            for param_group in optimizer_2.param_groups:  
                param_group['lr'] = lr_  
  
            if iter_num >= 800 and iter_num % 200 == 0:  
                model_1.eval()  
                model_2.eval()  
  
                if args.dataset_name == "Pancreas_CT":  
                    dice_sample_1 = test_patch.var_all_case(model_1,  
                                                            num_classes=num_classes,  
                                                            patch_size=patch_size,  
                                                            stride_xy=16,  
                                                            stride_z=16,  
                                                            dataset_name='Pancreas_CT')  
                    dice_sample_2 = test_patch.var_all_case(model_2,  
                                                            num_classes=num_classes,  
                                                            patch_size=patch_size,  
                                                            stride_xy=16,  
                                                            stride_z=16,  
                                                            dataset_name='Pancreas_CT')  
  
                elif args.dataset_name == "BraTS2019":  
                    dice_sample_1 = test_patch.var_all_case(model_1,  
                                                            num_classes=num_classes,  
                                                            patch_size=patch_size,  
                                                            stride_xy=64,  
                                                            stride_z=64,  
                                                            dataset_name='BraTS2019')
                    dice_sample_2 = test_patch.var_all_case(model_2,  
                                                            num_classes=num_classes,  
                                                            patch_size=patch_size,  
                                                            stride_xy=64,  
                                                            stride_z=64,  
                                                            dataset_name='BraTS2019')

                if dice_sample_1 > best_dice_1:  
                    best_dice_1 = dice_sample_1  
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model_1))  
                    torch.save(model_1.state_dict(), save_best_path)  
                if dice_sample_2 > best_dice_2:  
                    best_dice_2 = dice_sample_2  
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model_2))  
                    torch.save(model_2.state_dict(), save_best_path)  
  
                logging.info('Iteration %d - Model 1 Dice: %.3f (Best: %.3f)' % (iter_num, dice_sample_1, best_dice_1))  
                logging.info('Iteration %d - Model 2 Dice: %.3f (Best: %.3f)' % (iter_num, dice_sample_2, best_dice_2))  
  
                model_1.train()  
                model_2.train()  
  
            if iter_num >= max_iterations:  
                break  
        if iter_num >= max_iterations:  
            iterator.close()
            break  
  
    writer.close()
    
    logging.info('Training finished!')
    logging.info('Best Model 1 Dice: %.3f' % best_dice_1)
    logging.info('Best Model 2 Dice: %.3f' % best_dice_2)