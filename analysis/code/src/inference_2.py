import numpy as np
import cv2
import torch
import glob as glob
import os

from model import create_model
from config import ROOT, NUM_EPOCHS, COLOURS, CLASSES, BACKBONE, THRESHOLD

NUM_CLASSES = len(CLASSES)

# define the detection threshold...
#... any detection having score below this will be discarded
detection_threshold = THRESHOLD
print('-'*50)
print(f"Inferening on model backbone {BACKBONE} with detection_threshold = {detection_threshold}")
print('-'*50)

# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load model and trained weights
model = create_model(num_classes=NUM_CLASSES, backbone = BACKBONE).to(device)
model_path = os.path.join(ROOT, 'outputs', 'test', f'{BACKBONE}_model{NUM_EPOCHS}.pth')
model.load_state_dict(torch.load(
    model_path, map_location = device
))

model.eval()

# directory where all the images are present
DIR_TEST = os.path.join(ROOT, 'test_data')
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")




# Looping over the image paths and carrying out inference
for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i]).astype(np.float32)
    orig_image = image.copy()
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float64)
    # convert to tensor
    image = torch.tensor(image, dtype = torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image) # outputs will consit of two tensors [targets, images]
    
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to the detection threshold
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicted class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        # draw the bounding boxes and write class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (COLOURS[pred_classes[j]]), 3)
            cv2.putText(orig_image, pred_classes[j] + str(scores[j]), 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)
            
            # cv2.imshow('Prediction', orig_image)
            # cv2.waitKey(1)
            write_to_dir = os.path.join(ROOT, 'test_predictions', 'test', f'{image_name}.jpg')
            cv2.imwrite(write_to_dir, orig_image)
        print(f"Image {i+1} done...")
        print('-'*50)

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()