from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


def get_net(checkpoint_path=None):
    
    config = get_efficientdet_config('tf_efficientdet_d3')
    config.num_classes = 3
    config.image_size = (512,512)
    
    config.soft_nms = False
    config.max_det_per_image = 200
    
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        
    return DetBenchTrain(net)