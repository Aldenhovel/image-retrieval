from .Caption import CaptionInfer
from .YOLOv8 import YOLOv8Infer
from .Instructor import InstructorInfer
from .Topic import TopicInfer
import torch
import torch.nn as nn

class ImgToVector(nn.Module):
    
    def __init__(self):
        nn.Module.__init__(self)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.caption_model = CaptionInfer(self.device)
        self.yolo_model = YOLOv8Infer(self.device)
        self.instructor_model = InstructorInfer(self.device)
        self.topic_model = TopicInfer(self.device)
        
    def forward(self, img_path):
        caption = self.caption_model(img_path)
        objects = self.yolo_model(img_path)
        object_text = ' '
        for k, v in objects.items():
            object_text += f'{v} {k} , '
        object_text += " is in the photo . "
        mix_text = caption + object_text
        topic = self.topic_model(caption)
        return self.instructor_model(mix_text, intro=topic), mix_text
