# 1D-Grad-CAM
The implementation of [Grad-CAM](https://arxiv.org/pdf/1610.02391v1.pdf) for 1d data.

Original implementation in Keras : https://github.com/vense/keras-grad-cam

## Usage

```python3
import numpy as np
from keras.models import Model
from grad_cam import grad_cam

pred = model.predict(data_vector)
category_index = np.argmax(pred)
for layer in model.layers:
    if 'conv1d' in layer.name:
        conv_name = layer.name
heatmap = grad_cam(model, data_vector, category_index, conv_name, nb_classes)

```
### Arguments
-   data_vector : Input data (1D)
-   category_index : the index of predicted category
-   conv_name : the last convolutional layer of your model
-   nb_classes : the number of class
### Output
The vector of heatmap value (the same shape as data_vector)
