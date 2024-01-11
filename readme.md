# 图像检索系统

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com)

![framework](img/framework.svg)

<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="https://github.com/Aldenhovel/image-retrieval/blob/main/readme_en.md">English</a> 
    <p>
</h4>

## 简介

这是一个非常缝合的图像——文本检索框架，核心是通过对图像内容产生描述，再使用描述文本的相似性计算符合条件的图片。优点有：

- 即开即用，各个网络之间无需训练即可搭配使用。
- 在图像描述中可以手动加入自定义文本，使图像检索可以考虑在图像中无法识别的信息。
- 结合了图像字幕、目标检测、文本分类和文本嵌入模型，效果还可以。



## 模块

### 图像字幕模型

图像字幕模型负责对图像内容产生文本描述，在框架中发挥最主要作用。这里使用了[noamrot/FuseCap_Image_Captioning](https://hf-mirror.com/noamrot/FuseCap_Image_Captioning)模型，了解更多：

>
>
>Rotstein, Noam, et al. "FuseCap: Leveraging Large Language Models to Fuse Visual Data into Enriched Image Captions." *arXiv preprint arXiv:2305.17718* (2023).

### 目标检测模型

目标检测模型负责检测图像中存在的目标物体，在框架中对产生的图像字幕进行补充，主要是强调物体的数量关系，防止图像字幕遗漏元素或者搞错数目。这里使用了[ultralytics/YOLOv8](https://github.com/ultralytics/ultralytics)模型，了解更多：

>
>
>Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

### 文本分类模型

文本分类模型负责对图像字幕和目标检测所产生的图像综合描述内容进行分类，所产生的文本类别会参与后续文本嵌入中作为引导线索，提升文本嵌入的精确度。这里使用了[cardiffnlp/tweet-topic-21-multi](https://hf-mirror.com/cardiffnlp/tweet-topic-21-multi)模型，可以将文本进行21个类别的判别，了解更多：

>
>
>Antypas, Dimosthenis, et al. "Twitter topic classification." *arXiv preprint arXiv:2209.09824* (2022).

### 文本嵌入模型

文本嵌入模型负责将文本转换为特征向量，是计算文本相似性的最重要环节。这里使用了[hkunlp/instructor-large](https://hf-mirror.com/hkunlp/instructor-large)模型，依据instruction和sentence两个参数，在本框架中将图像综合描述作为sentence，将文本类别作为introduction进行编码计算得到特征向量。了解更多：

>
>
>Su, Hongjin, et al. "One embedder, any task: Instruction-finetuned text embeddings." *arXiv preprint arXiv:2212.09741* (2022).



## 环境

```
torch==1.13.1
numpy==1.21.6
matplotlib==3.5.3
transformers==4.31.0
scipy==1.7.3
ultralytics==8.0.228
tqdm==4.66.1
```

由于文件大小限制，还需要手动将HuggingFace的模型`bin`文件下载移动到`cardiffnlp/tweet-topic-21-multi/`和`noamrot/FuseCap/`两个对应目录：

- [tweet-topic-21-multi](https://hf-mirror.com/cardiffnlp/tweet-topic-21-multi/blob/main/pytorch_model.bin)
- [fusecap](https://hf-mirror.com/noamrot/FuseCap_Image_Captioning/blob/main/pytorch_model.bin)



## 使用

**检查模块可用性** 使用`test_utils.ipynb`可以检查各个模型是否已经正确部署。

**建立图像特征索引** 使用`build_index.ipynb`可以对`gallery/`内的`.jpg`图像文件搭建特征索引，结果会保存至`tmp/`中。

**图像检索** 使用`retrieval_image.ipynb`可以检索图像（需要先建立图像特征索引）。



## 示例

```
a plane flying in the sky .
```

![exam0](img/example0.png)

```
a bus in the street .
```

![exam1](img/example1.png)



## 实验数据

- 在Filckr8k上的图像检索实验

  |                               |  @1   |  @5   |  @20  | avg/1k |
  | :---------------------------: | :---: | :---: | :---: | :----: |
  |      YOLOv8 + Instructor      | 14.9% | 33.6% | 52.4% | 66.95  |
  |     FuseCap + Instructor      | 39.7% | 67.3% | 87.3% | 12.53  |
  | FuseCap + YOLOv8 + Instructor | 43.2% | 69.6% | 89.2% | 10.76  |

- 在Flickr30k上的图像检索实验

  |                               |  @1   |  @5   |  @20  | avg/1k |
  | :---------------------------: | :---: | :---: | :---: | :----: |
  |      YOLOv8 + Instructor      | 15.4% | 31.6% | 50.7% | 85.34  |
  |     FuseCap + Instructor      | 42.8% | 70.2% | 86.3% | 16.27  |
  | FuseCap + YOLOv8 + Instructor | 47.6% | 74.0% | 89.0% | 11.54  |

- 在MSCOCO上的图像检索实验

  |                               |  @1   |  @5   |  @20  | avg/1k |
  | :---------------------------: | :---: | :---: | :---: | :----: |
  |      YOLOv8 + Instructor      | 12.7% | 25.4% | 45.8% | 120.35 |
  |     FuseCap + Instructor      | 34.1% | 59.4% | 80.4% | 15.03  |
  | FuseCap + YOLOv8 + Instructor | 37.9% | 63.7% | 83.7% | 12.88  |



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=aldenhovel/image-retrieval&type=Date)](https://star-history.com/#aldenhovel/image-retrieval&Date)

