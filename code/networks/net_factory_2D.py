def net_factory(net_type="unet", in_chns=3, class_num=4, args=None):

    if net_type == "resunet_feature":
        from networks.resunet import UNet_DS
        net = UNet_DS(in_chns=in_chns, class_num=class_num)
    elif net_type == "swinunet_feature":
        from networks.vision_transformer import SwinUnet_DS as ViT_seg
        from networks.config import get_config
        config = get_config(args)
        net = ViT_seg(config, img_size=224, num_classes=class_num)
        net.load_from(config)
    else:
        print("error model")
        exit()
    return net