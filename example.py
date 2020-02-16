pred = model.predict(data_vector)
category_index = [np.argmax(i) for i in pred]
for layer in model.layers:
    if 'conv1d' in layer.name:
        conv_name = layer.name
tmp = np.reshape(x_eval[i1], (1, x_eval[i1].shape[0], x_eval[i1].shape[1]))
tmp = grad_cam(model, tmp, category_index[i1], conv_name, nb_classes)
