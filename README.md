# Garbage Classification
Google ML Winter Camp 2020 Project  
Team Anonymous: Zhibo Wang, Xiaoyu Chen, Huihan Yao

## Description
Our app can detect and classify multiple garbage instances in one image. The backend consist of a single-class(entity) object detection module and an image classification module. Firstly a pretrained object detection model SSDLite detect all possible objects from the given image. Then we select top-3 objects with the largest detection confidence to do image classification. The image is cropped according to detected bounding boxes padded with a margin before it is put into MobileNetV2 to do classification. Considering the given data “garbage-classification” don’t have bounding boxes for objects, we didn’t train a single object detection model for one-step detection and classification.

## Performance
We used “garbage-classification” dataset to train our classification model. Our model reached a best validation accuracy of 85% at epoch 36. On our devices, the typical time of object detection is ~40ms, and image classification costs ~50ms for each object detected.

## Reference
- Garbage Classification Dataset. gs://ml-camp/garbage_classify
- TensorflowLite in Android APP. https://github.com/tensorflow/examples/tree/master/lite/examples
- Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. MobileNetV2: Inverted Residuals and Linear Bottlenecks. arXiv preprint 1801.04381.
