# Knowledge Distillation

Knowledge distillation is a technique where we use a large pretained model, called the teacher, to train a smaller model, called the student. We aim to make a student model which has faster inference times than the teacher without losing out on too much accuracy. This technique can be used to quickly train smaller models for specialised tasks, even when we do not have a lot of training data.

The CIFAR code uses a ResNet50 model (with the classification head changed to predict 10 classes) as the teacher and a ResNet18 model as the student.

To learn more about Knowledge distillation, refer to Report.pdf

Most of the code used in the CIFAR experiments was adapted from [this repository](https://github.com/wangyz1608/knowledge-distillation-via-nd).
