from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import expit

import torch.nn as nn

class TopicInfer(nn.Module):
    
    def __init__(self, device):
        nn.Module.__init__(self)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("./models/cardiffnlp/tweet-topic-21-multi")
        self.model = AutoModelForSequenceClassification.from_pretrained("./models/cardiffnlp/tweet-topic-21-multi").to(self.device)
        self.class_mapping = self.model.config.id2label
        
        
    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors='pt').to(self.device)
        output = self.model(**tokens)
        scores = output[0][0].detach().cpu().numpy()
        scores = expit(scores)
        predictions = (scores >= 0.2) * 1
        topic = ''
        for i in range(len(predictions)):
            if predictions[i]:
                topic += self.class_mapping[i].replace('_&_', ' and ').replace('_', ' ')
        topic = f'represent the {topic} title:'
        return topic