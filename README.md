# ImageClassification

Python2.7 - Keras (using TensorFlow backend) code for Image Classification task.

By default, this code uses ResNet50 trained on ImageNet as a pretrained model from the link below
https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5


## Training on Custom Data

1. Data Structure<br /><br />
When training on your custom data, you should construct your data format - you might want to choose one of two options.

(1) with annotation file (see data/example_annotation)

```
[data]
... [Your_Custom_Data]
...... annotation.txt
...... [Images]
......... a.jpg
......... b.jpg
```

where annotation.txt's format should follow:
Images/a.jpg  (tab) label_a
Images/b.jpg  (tab) label_b

(2) with no annotation file (see data/example_categorical_folders)

```
[data]
... [Your_Custom_Data]
...... [label_a]
......... a1.jpg
......... a2.jpg
...... [label_b]
......... b1.jpg
......... b2.jpg
```
<br />
2. Class Number<br /><br />
If your class number is different from that of ImageNet (i.e., your_class_num is not 1,000), the last Dense layer will automatically be replaced.

<br />

3. Classification / Regression Mode<br /><br />
By default, the code will train a classification model.<br />
You can train a regression model by adding "--mode regression" in run_train.sh
