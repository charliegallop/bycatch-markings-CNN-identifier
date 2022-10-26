import torchvision
from config import DEVICE
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes, backbone, load_state_from = None):

    if backbone == 'resnet':
        # load Faster RCNN pre-trained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights = 'DEFAULT')

        # get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    # if backbone == 'mobilenet':
    #      # load Faster RCNN pre-trained model
    #     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

    #     # get the number of input features
    #     in_features = model.roi_heads.box_predictor.cls_score.in_features
    #     # define a new head for the detector with required number of classes
    #     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if backbone == 'mobilenet':
         # load Faster RCNN pre-trained model
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained = True)

        # get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if backbone == 'mobilenet_320':
         # load Faster RCNN pre-trained model
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained = True)

        # get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # if load_state_from:
    #     model = create_model(num_classes=num_classes, backbone = backbone).to(DEVICE)
    #     model_path = load_state_from
    #     model.load_state_dict(torchvision.load(
    #         model_path, map_location = DEVICE
    #     ))

    return model