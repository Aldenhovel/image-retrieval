import os
os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO
from PIL import Image
from collections import defaultdict

import torch.nn as nn

class YOLOv8Infer(nn.Module):
    
    def __init__(self, device):
        nn.Module.__init__(self)
        self.device = device
        self.model = YOLO("./models/yolov8m-oiv7.pt").to(device)
        
    def forward(self, img_path):
        im1 = Image.open(img_path)
        result = self.model.predict(source=im1, save=False)[0]
        names = result.names
        clss = result.boxes.cls.detach().cpu().numpy().astype('int')
        entities = defaultdict(int)
        for c in clss:
            entities[names[c].lower()] += 1
        nums = ['no', 'one',  'two', 'three', 'four']
        for word, count in entities.items():
            entities[word.lower()] = nums[count] if count < len(nums) else 'many'
        return entities