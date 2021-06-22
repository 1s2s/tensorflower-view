# -*- coding: utf-8 -*
import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
                                       

MODEL_NAME = 'cnn'
pathfile = '.\\cnn_result\\'
# Freeze the graph

input_graph_path = pathfile+'tfdroid.pbtxt'
checkpoint_path = pathfile+'tfdroid'+'.ckpt'
input_saver_def_path = ""
input_binary = False
output_node_names = "accuracy"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")



# Optimize for inference

input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["x"], # an array of the input nodXBe(s)
        ["accuracy"], # an array of output nodes
        tf.float32.as_datatype_enum)

# Save the optimized graph

with tf.gfile.FastGFile(output_optimized_graph_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())

# tf.train.write_graph(output_graph_def, './', output_optimized_graph_name)
