import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow as tf
import argparse
from utils import helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--frontend', type=str, default=None, required=True, help='The frontend you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
parser.add_argument('--export_path', type=str, default=None, required=True, help='The path to export the model to.')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin export *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Frontend -->", args.frontend)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Export path -->", args.export_path)

# Initializing network
sess=tf.Session()

net_input = tf.placeholder(tf.float32,shape=[None,args.crop_width,args.crop_height,3])
net_output = tf.placeholder(tf.float32,shape=[None,args.crop_width,args.crop_height,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input=net_input,
                                        frontend=args.frontend,
                                        num_classes=num_classes,
                                        crop_width=args.crop_width,
                                        crop_height=args.crop_height,
                                        is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

input_dict = {"input": net_input}
output_dict = {"output": network}

print("Saving to", args.export_path)
tf.saved_model.simple_save(sess, args.export_path, inputs=input_dict, outputs=output_dict)

print("Done")