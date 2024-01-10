from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity

import torch.nn as nn

class InstructorInfer(nn.Module):
    
    def __init__(self, device):
        nn.Module.__init__(self)
        self.device = device
        self.model = INSTRUCTOR('hkunlp/instructor-large').to(device)
        
    def forward(self, x, intro=""):
        return self.model.encode([[intro, x]])