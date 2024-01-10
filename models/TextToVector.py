from .Topic import TopicInfer
from .Instructor import InstructorInfer
import torch
import torch.nn as nn

class TextToVector(nn.Module):
    
    def __init__(self):
        nn.Module.__init__(self)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.topic_model = TopicInfer(self.device)
        self.instructor_model = InstructorInfer(self.device)
        
    def forward(self, text):
        topic = self.topic_model(text)
        vector = self.instructor_model(text, intro=topic)
        return vector