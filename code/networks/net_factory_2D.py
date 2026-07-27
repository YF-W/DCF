def net_factory(net_type="unet", in_chns=3, class_num=4, args=None):

    if net_type == "nnunet":
        from networks.nnunet import initialize_network
        net = initialize_network(threeD=False, num_classes=class_num)
    elif net_type == "unet":
        from networks.unet import UNet
        net = UNet(in_chns=in_chns, class_num=class_num)
    elif net_type == "unet_small":
        from networks.unet_2 import UNet_DS as UNet_DS_small
        net = UNet_DS_small(in_chns=in_chns, class_num=class_num)
    elif net_type == "deeplabv3+":
        from networks.deeplabv3_plus_2 import DeepLab
        from networks.modeling import deeplabv3plus_resnet50
        net = DeepLab(num_classes=class_num)
    elif net_type == "deeplabv3+_feature":
        from networks.deeplabv3_plus_2 import DeepLab_DS
        from networks.modeling import deeplabv3plus_resnet50
        net = DeepLab_DS(num_classes=class_num)
    elif net_type == "transunet":
        from networks.TransUNet import TransUNet
        net = TransUNet(num_classes=class_num)
    elif net_type == "swinunet":
        from networks.vision_transformer import SwinUnet as ViT_seg
        from networks.config import get_config
        config = get_config(args)
        net = ViT_seg(config, img_size=224, num_classes=class_num)
        net.load_from(config)
    elif net_type == "resunet":
        from networks.resunet import UNet
        net = UNet(in_chns=in_chns, class_num=class_num)
    elif net_type == "resunet_feature":
        from networks.resunet import UNet_DS
        net = UNet_DS(in_chns=in_chns, class_num=class_num)
    elif net_type == "unet_feature":
        from networks.unet import UNet_DS
        net = UNet_DS(in_chns=in_chns, class_num=class_num)
    elif net_type == "swinunet_feature":
        from networks.vision_transformer import SwinUnet_DS as ViT_seg
        from networks.config import get_config
        config = get_config(args)
        net = ViT_seg(config, img_size=224, num_classes=class_num)
        net.load_from(config)
    elif net_type == "efficientnet_feature":
        from networks.unet_efficientnet import UNet_DS
        net = UNet_DS(in_chns=in_chns, class_num=class_num)
    elif net_type == "repvit_feature":
        from networks.unet_repvit import UNet_DS
        net = UNet_DS(in_chns=in_chns, class_num=class_num)

    else:
        print("error model")
        exit()
    return net