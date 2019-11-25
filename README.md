# Multi-instance CNN for Breast Cancer Classification

Note: Project was part of a Praktikum at the Technical University of Munich. Credit goes also to Hannes Hase who worked in the project.

## Problem Description

The dataset comes from the [BACH breast cancer histology images challenge] (https://iciar2018-challenge.grand-challenge.org/Dataset/). The focus here is on the classification subtask, which is a multi-class classification task.

The dataset contains four classes ordered by the predominant cancer type, each class has 100 examples and images have a size of 2048 x 1536 pixels. A sample from each class is shown in the following figure.

![Sample images from each class](https://github.com/omsh/Multi-instance-CNN-for-medical-imaging/tree/master/imgs/sample_images.png "Figure: Sample image from each class")

## Overview of the Solution Used

Several patches are extracted from each image and fed to a deep CNN as a bag of images. Afterwards, multi-instance learning is employed by implementing a custom pooling layer that pools over the feature dimension of the final image representation before feeding it to a classifier (e.g. a couple of feed-forward layers).

The implementation framework is adopted from [Tensorflow Project Template](https://github.com/MrGemy95/Tensorflow-Project-Template) with some customizations. The following figure shows the high-level structure of the project.

![High-level structure of the project](https://github.com/omsh/Multi-instance-CNN-for-medical-imaging/tree/master/imgs/project_structure.png "Figure: High-level structure of the project")

An overview of the main model architecture contains a deep CNN followed by two branches (single-instance and multi-instance), where each has its own loss, with the multi-instance branch containing the custom pooling layer. The final loss is a weighted combination of both losses. The following figure shows an overview of the model architecture. This model is adopted from the paper [1] (https://github.com/omsh/Multi-instance-CNN-for-medical-imaging#References).

![Overview of the model architecture](https://github.com/omsh/Multi-instance-CNN-for-medical-imaging/tree/master/imgs/model_arch.png "Figure: Overview of the model architecture")

A couple of different CNNs were tested as backbone, with most of the results reported on the famous ResNet [2] (https://github.com/omsh/Multi-instance-CNN-for-medical-imaging#References)

The custom pooling layer can perform one of three operations; average-pooling, max-pooling, or log-sum-exponent. Results for experiements with the three pooling functions are reported.

The combined weighted loss can also be varied during training, as done in [1] (https://github.com/omsh/Multi-instance-CNN-for-medical-imaging#References). The following figure shows the weights of both losses (single-instance and multi-instance).

![Variable loss weights](https://github.com/omsh/Multi-instance-CNN-for-medical-imaging/tree/master/imgs/var_loss_weights.png "Figure: Variable loss weights")


## References

1. Conjeti S., Paschali M., Katouzian A., Navab N. (2017) Deep Multiple Instance Hashing for Scalable Medical Image Retrieval. In: Descoteaux M., Maier-Hein L., Franz A., Jannin P., Collins D., Duchesne S. (eds) Medical Image Computing and Computer-Assisted Intervention − MICCAI 2017. MICCAI 2017. Lecture Notes in Computer Science, vol 10435.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).







