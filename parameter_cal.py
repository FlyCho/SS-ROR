# from tensorflow.python import pywrap_tensorflow
# import os
# import numpy as np
# model_dir = "./checkpoints/"
# checkpoint_path = os.path.join(model_dir, "model.ckpt-37500")
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# total_parameters = 0
# for key in var_to_shape_map:#list the keys of the model
#     # print(key)
#     # print(reader.get_tensor(key))
#     shape = np.shape(reader.get_tensor(key))  #get the shape of the tensor in the model
#     shape = list(shape)
#     # print(shape)
#     # print(len(shape))
#     variable_parameters = 1
#     for dim in shape:
#         # print(dim)
#         variable_parameters *= dim
#     # print(variable_parameters)
#     total_parameters += variable_parameters
#
# print(total_parameters)

import tensorflow as tf
def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))