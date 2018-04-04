# ImageClassification

Python2.7 - Keras (using TensorFlow backend) code for Image Classification task.

By default, this code uses ResNet50 trained on ImageNet as a pretrained model from the link below
https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5

When training on your custom data, you should construct your data format - you might want to choose one of two options.

(1) with annotation file (see data/example_annotation)

```
[data]<br />
... [Your_Custom_Data]<br />
...... annotation.txt<br />
...... [Images]<br />
......... a.jpg<br />
......... b.jpg<br />
```

where annotation.txt's format should follow:
Images/a.jpg  (tab) label_a
Images/b.jpg  (tab) label_b

(2) with no annotation file (see data/example_categorical_folders)

```
[data]<br />
... [Your_Custom_Data]<br />
...... [label_a]<br />
......... a1.jpg<br />
......... a2.jpg<br />
...... [label_b]<br />
......... b1.jpg<br />
......... b2.jpg<br />
```

If your class number is different from that of ImageNet (i.e., your_class_num is not 1,000), the last Dense layer will automatically be replaced.

You can train a regression model by adding "--mode regression" in run_train.sh
