def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2, shape=None):
    if net_type == "unet_3d":
        from networks.unet_3D import unet_3D
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "attention_unet":
        from networks.attention_unet import Attention_UNet
        net = Attention_UNet(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "voxresnet":
        from networks.VoxResNet import VoxResNet
        net = VoxResNet(in_chns=in_chns, feature_chns=64,
                        class_num=class_num).cuda()
    elif net_type == "vnet":
        from networks.vnet import VNet
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "nnunet":
        from networks.nnunet import initialize_network
        net = initialize_network(num_classes=class_num).cuda()
    elif net_type == "unetr":
        from networks.unetr import UNETR
        net = UNETR(img_shape=shape, input_dim=in_chns, output_dim=class_num).cuda()
    elif net_type == "vnet_bfa":
        from networks.VNet_BFA import VNet_BFA
        net = VNet_BFA(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()

    else:
        net = None
    return net