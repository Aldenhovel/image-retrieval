import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import BlipProcessor, BlipForConditionalGeneration

class CaptionInfer(nn.Module):
    
    def __init__(self, device):
        nn.Module.__init__(self)
        self.device = device
        self._fusecap_processor = BlipProcessor.from_pretrained("./models/noamrot/FuseCap")
        self._fusecap_model = BlipForConditionalGeneration.from_pretrained("./models/noamrot/FuseCap").to(self.device)
    
    def forward(self, img_path):
        raw_image = Image.open(img_path).convert('RGB')
        text = "a picture of "
        inputs = self._fusecap_processor(raw_image, text, return_tensors="pt").to(self.device)
        inputs['max_new_tokens'] = 60
        out = self._fusecap_model.generate(**inputs, num_beams = 3)
        caption = self._fusecap_processor.decode(out[0], skip_special_tokens=True)
        return caption