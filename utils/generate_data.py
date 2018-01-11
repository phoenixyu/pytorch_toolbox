from preprocess import write_train_list,crop,generate_trainval_list
from preprocess import convert_label_to_vis,convert_vis_to_label
import os
import skimage.io as io
import numpy as np

ProjectDir = "/home/dl/phoenix_lzx/torch/data"
crop_size = 320
training_data_stage1_dir = os.path.join(ProjectDir,"seaship")

img_file_list = set()
for item in os.listdir(training_data_stage1_dir):
    img_file_list.add(item.split(".")[0])
img_file_list = list(img_file_list)

def generate_stat(label_file_lists):
	label_list=[]
	for label_file in label_file_lists:
		label = io.imread(label_file)
		label_list = label_list + label.flatten().tolist()
	count_label = np.bincount(label_list)
	return count_label

def generate_dataset(dataset_dir,crop_size,img_list,label_list):
	img_path=os.path.join(dataset_dir,'img')
	label_path =os.path.join(dataset_dir,'label')
	visualize_gt_path = os.path.join(dataset_dir,'visualize_gt')

	if(not os.path.exists(img_path)):
		os.mkdir(img_path)
	if(not os.path.exists(label_path)):
		os.mkdir(label_path)
	if(not os.path.exists(visualize_gt_path)):
		os.mkdir(visualize_gt_path)
	for i in range(len(img_list)):
		crop(img_list[i],label_list[i],crop_size,crop_size,prefix='%d'%(i+1),save_dir=dataset_dir,crop_label=True)
	generate_trainval_list(dataset_dir)
	write_train_list(dataset_dir)

##save crf label
#convert_vis_to_label(os.path.join(training_data_stage1_dir,'1_visual_crf.png'),os.path.join(training_data_stage1_dir,'1_class_crf.png'))
#convert_vis_to_label(os.path.join(training_data_stage1_dir,'2_visual_crf.png'),os.path.join(training_data_stage1_dir,'2_class_crf.png'))

img_list_1=[os.path.join(training_data_stage1_dir,'{}.jpg'.format(item)) for item in img_file_list]
label_list_1=[os.path.join(training_data_stage1_dir,'{}.png'.format(item)) for item in img_file_list]

# dataset
# stat = generate_stat(label_list_1)
# print("dataset rate: ",np.array(stat)*1.0/np.min(stat[np.nonzero(stat)]))
dataset_dir=os.path.join(ProjectDir,"dataset/seaship-train")
if(not os.path.exists(dataset_dir)):
	os.mkdir(dataset_dir)
	generate_dataset(dataset_dir,crop_size,img_list_1,label_list_1)
	print("create dataset...")
else:
    print("dataset exists, pass!")
