import numpy as np
from models.Topic import TopicInfer
from models.Instructor import InstructorInfer
import torch
import matplotlib.pyplot as plt


class ImgSelector:
    
    def __init__(self, imgs, mtx):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.topic_model = TopicInfer(self.device)
        self.instructor_model = InstructorInfer(self.device)
        self.imgs = imgs
        self.mtx = mtx
        
    def search(self, text, k=5):
        t = self.topic_model(text)
        v = self.instructor_model(text, intro=t).T
        scores = np.dot(self.mtx, v).T[0]
        top_k_indices = self.topk(scores, k)
        sim = scores[top_k_indices]
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4), tight_layout=True)
        for ix, indice in enumerate(top_k_indices):
            row, col = ix // 5, ix % 5
            image = plt.imread(self.imgs[indice])
            axes[row, col].imshow(image)
            axes[row, col].set_title(f'[{ix}]   {sim[ix]:.3f}')
            axes[row, col].axis('off') 
        plt.show()
        
    
    def topk(self, scores, k):
        sorted_indices = np.argsort(-scores)
        top_k_indices = sorted_indices[:k]
        return top_k_indices
        