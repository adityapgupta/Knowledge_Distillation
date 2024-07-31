# Knowledge Distillation
Knowledge distillation is a technique where we use a large pretained model, called the teacher, to train a smaller model, called the student. We aim to make a student model which has faster inference times than the teacher without losing out on too much accuracy. This technique can be used to quickly train smaller models with specialised tasks, even when we do not have a lot of training data.

The CIFAR code uses a pretrained ResNet50 model as the teacher. Download the file resnet50.pt from the [here](https://drive.google.com/drive/folders/196Kj-J8N4xqRb66OuJ5eSmqF0axfGK6W?usp=drive_link) and place it in CIFAR/Models to use the ResNet50.


To learn more about Knowledge distillation, refer to Report.pdf

Most of the code used in the CIFAR experiments was adapted from [this repository](https://github.com/wangyz1608/knowledge-distillation-via-nd).
