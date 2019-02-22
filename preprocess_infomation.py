"""
Before running this code make sure that 
1) Images have been segregated into train,test and validation 
2) Pycocotool have been installed 

3) SELECT MODE 
ATTENTION/ WITHOUT ATTTENTION

Modify the paths to local paths 

"""

import cPickle
import os
import numpy as np
import re
from utilities import log
from nltk.tokenize import word_tokenize
import sys
from pycocotools.coco import COCO

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile


############################################################
# CHANGE THE VALUES DEFINED HERE
PYCOCOTOOLS_PATH ="/home/cul10kk/ML_project/coco/PythonAPI"
MODE= 0   ##### Mode 0 - no attention , 1- attention
#CNN model
model_dir = "/home/cul10kk/CS224n_project/inception"

############################################################

sys.path.append(PYCOCOTOOLS_PATH)


# DIRECTORIES
captions_dir = "coco/annotations/"
data_dir = "coco/data/"
val_img_dir = "coco/images/val/"
test_img_dir = "coco/images/test/"
train_img_dir = "coco/images/train/"

########################################################
#PREPROCESS THE VALIDATION AND TEST IMAGES
########################################################
# image locations

val_img_paths = [val_img_dir + file_name for file_name in\
                 os.listdir(val_img_dir) if ".jpg" in file_name]
test_img_paths = [test_img_dir + file_name for file_name in\
                  os.listdir(test_img_dir) if ".jpg" in file_name]
train_img_paths = [train_img_dir + file_name for file_name in\
                       os.listdir(train_img_dir) if ".jpg" in file_name]

val_img_ids = np.array([])
for val_img_path in val_img_paths:
    img_name = val_img_path.split("/")[3]
    img_id = img_name.split("_")[2].split(".")[0].lstrip("0")
    img_id = int(img_id)
    val_img_ids = np.append(val_img_ids, img_id)
test_img_ids = np.array([])
for test_img_path in test_img_paths:
    img_name = test_img_path.split("/")[3]
    img_id = img_name.split("_")[2].split(".")[0].lstrip("0")
    img_id = int(img_id)
    test_img_ids = np.append(test_img_ids, img_id)
#cPickle.dump(val_img_ids, open(os.path.join("coco/data/", "val_img_ids"), "wb"))
#cPickle.dump(test_img_ids, open(os.path.join("coco/data/", "test_img_ids"), "wb"))
##########################################################
##########################################################
#Preprocess the caption information
##########################################################
# val_img_ids and test_img_ids RESUED
##########################################################

word_counts = {} 
vocabulary = []
  # DEFINING THE VOCABULARY
    captions_file = "coco/annotations/captions_train2014.json" 
    coco = COCO(captions_file)
    print("Defining the vocabulary")
    for sentence in tqdm(coco.all_captions()):
        for w in word_tokenize(sentence.lower()):
            word_counts[w] = word_counts.get(w, 0) + 1.0

# get the captions for training and validation 
def get_captions(type_of_data):
    captions_file = "coco/annotations/captions_%s2014.json" % type_of_data

    coco = COCO(captions_file)

    # extract the information for the coco API
    img_ids    = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns] 
    captions   = [coco.anns[ann_id]['caption']  for ann_id in coco.anns]
    caption_id = [coco.anns[ann_id]['id']       for ann_id in coco.anns]
    caption_id_2_img_id = img_ids
    
    if type_of_data == "train":
        train_caption_id_2_caption = dict(zip(caption_id,captions))
    elif type_of_data == "val":
        # validation and test need to be handled sparately
        if img_id in test_img_ids:
           test_caption_id_2_caption[caption_id] = caption
        elif img_id in val_img_ids:
           val_caption_id_2_caption[caption_id] = caption

train_caption_id_2_caption = {}
test_caption_id_2_caption = {}
val_caption_id_2_caption = {}
caption_id_2_img_id = {}
get_captions("train")
get_captions("val")

cPickle.dump(caption_id_2_img_id,
        open(os.path.join(data_dir, "caption_id_2_img_id"), "wb"))

pretrained_words = []
word_vectors = []
with open(os.path.join(captions_dir, "glove.6B.300d.txt")) as file:
    for line in file:
        line_elements = line.split(" ")
        #word
        word = line_elements[0]
        #get the word vector
        word_vector = line_elements[1:]
        # store
        pretrained_words.append(word)
        word_vectors.append(word_vector)


# get all words that have a pretrained word embedding:
vocabulary = []
for word in word_counts:
    word_count = word_counts[word]
    if word_count >= 5 and word in pretrained_words:
        vocabulary.append(word)

# add "<SOS>", "<UNK>" and "<EOS>" to the vocabulary:
vocabulary.insert(0, "<EOS>")
vocabulary.insert(0, "<UNK>")
vocabulary.insert(0, "<SOS>")


#cPickle.dump(vocabulary,
#        open(os.path.join(data_dir, "vocabulary"), "wb"))

#replaces words that are not in the vocabulary and appends the <Start> <end>
ef fix_tokenize(self,caption_dict,name)
	print("Tokenizing %s"%name)
	for caption_id,caption in tqdm(caption_dict.items()):
    	# prepend the caption with an <SOS> token;
    	tokenized_caption = []
    	caption.insert(0, "<SOS>")
    	caption.append("<EOS>")
    	for word_index in range(len(caption)):
    	    word = caption[word_index]
    	    if word not in vocabulary:
				word = "<UNK>"
    	        caption[word_index] = word
        	word_id = vocabulary.index(word)
    	# append the caption with an <EOS> token:
        tokenized_caption.append(word_id)
    	tokenized_caption = np.array(tokenized_caption)
		if(name == "train"):
    		train_caption_id_2_caption[caption_id] = tokenized_caption

    	caption_length = len(caption)
    	if caption_length not in train_caption_length_2_caption_ids:
    	    train_caption_length_2_caption_ids[caption_length] = [caption_id]
    	else:
    	    train_caption_length_2_caption_ids[caption_length].append(caption_id)
    	no_of_captions = len(caption_ids)
    	train_caption_length_2_no_of_captions[caption_length] = no_of_captions
	log("Tokenized %s"%name)
	
fix_tokenize(train_caption_id_2_caption,"train")
fix_tokenize(val_caption_id_2_captioni ,"val")
fix_tokenize(test_caption_id_2_caption ,"test")

cPickle.dump(train_caption_id_2_caption, open(os.path.join(data_dir,
        "train_caption_id_2_caption"), "wb"))
cPickle.dump(test_caption_id_2_caption, open(os.path.join(data_dir,
        "test_caption_id_2_caption"), "wb"))
cPickle.dump(val_caption_id_2_caption, open(os.path.join(data_dir,
        "val_caption_id_2_caption"), "wb"))


cPickle.dump(train_caption_length_2_no_of_captions,
        open(os.path.join("coco/data/",
        "train_caption_length_2_no_of_captions"), "wb"))
cPickle.dump(train_caption_length_2_caption_ids,
        open(os.path.join("coco/data/",
        "train_caption_length_2_caption_ids"), "wb"))

##########################################################
##########################################################
#Preprocess the caption information
##########################################################
# Reuses
# vocabulary
# word_vector
##########################################################
word_vec_dim = 300
vocab_size = len(vocabulary)

pretrained_words = []
word_vectors = []
with open(os.path.join(captions_dir, "glove.6B.300d.txt")) as file:
    for line in file:
        # remove the new line char at the end:
        line = line.strip()

        # seperate the word from the word vector:
        line_elements = line.split(" ")
        word = line_elements[0]
        word_vector = line_elements[1:]

        # save:
        pretrained_words.append(word)
        word_vectors.append(word_vector)
# create an embedding matrix where each row is the pretrained word vector
embeddings_matrix = np.zeros((vocab_size, word_vec_dim))
for vocab_index, word in enumerate(vocabulary):
    if vocab_index % 1000 == 0:
        print vocab_index
        log(str(vocab_index))

    if word not in ["<SOS>", "<UNK>", "<EOS>"]: # (the special tokens are initialized with zero vectors)
        word_embedd_index = pretrained_words.index(word)
        word_vector = word_vectors[word_embedd_index]
        # convert into a numpy array:
        word_vector = np.array(word_vector)
        # convert everything to floats:
        word_vector = word_vector.astype(float)
        # add to the matrix:
        embeddings_matrix[vocab_index, :] = word_vector

# save the embeddings_matrix to disk:
cPickle.dump(embeddings_matrix,
        open(os.path.join(data_dir, "embeddings_matrix"), "wb"))
##########################################################
##########################################################
# RUN A PRETAINED CNN to get the image output
##########################################################

##########################################################
# WITHOUT ATTENTION
##########################################################
def load_pretrained_CNN():
    # define where the pretrained inception model is located:
    path_to_saved_model = os.path.join(model_dir,
            "classify_image_graph_def.pb")
    with gfile.FastGFile(path_to_saved_model, "rb") as model_file:
        # create an empty GraphDef object:
        graph_def = tf.GraphDef()

        # import the model definitions:
        graph_def.ParseFromString(model_file.read())
        _ = tf.import_graph_def(graph_def, name="")

def extract_img_features(img_paths, demo=False):
    img_id_2_feature_vector = {}
    # load the Inception-V3 model:
    load_pretrained_CNN()

    with tf.Session() as sess:
        # get the second-to-last layer in the Inception-V3 model (this
        # is what we will use as a feature vector for each image):
        second_to_last_tensor = sess.graph.get_tensor_by_name("pool_3:0")

        for step, img_path in enumerate(img_paths):
            if step % 100 == 0:
                print step
                log(str(step))

            # read the image:
            img_data = gfile.FastGFile(img_path, "rb").read()
            try:
                # get the img's corresponding feature vector:
                feature_vector = sess.run(second_to_last_tensor,
                        feed_dict={"DecodeJpeg/contents:0": img_data})
            except:
                print "JPEG error for:"
                print img_path
                print "******************"
                log("JPEG error for:")
                log(img_path)
                log("******************")
            else:
                # # flatten the features to an np.array:
                feature_vector = np.squeeze(feature_vector)

                if not demo:
                    # get the image id:
                    img_name = img_path.split("/")[3]
                    img_id = img_name.split("_")[2].split(".")[0].lstrip("0")
                    img_id = int(img_id)
                else: # (if demo:)
                    # we're only extracting features for one img, (arbitrarily)
                    # set the img id to 0:
                    img_id = 0

                # save the feature vector and the img id:
                img_id_2_feature_vector[img_id] = feature_vector

        return img_id_2_feature_vector

##########################################################

##########################################################
# WITH ATTENTION
##########################################################
# reuses 
##########################################################

def extract_img_features_attention(img_paths, demo=False):

    # load the Inception-V3 model:
    load_pretrained_CNN()

    # load the parameters for the feature vector transform:
    transform_params = cPickle.load(open("coco/data/img_features_attention/transform_params/numpy_params"))
    W_img = transform_params["W_img"]
    b_img = transform_params["b_img"]

    with tf.Session() as sess:
        # get the third-to-last layer in the Inception-V3 model (a tensor
        # of shape (1, 8, 8, 2048)):
        img_features_tensor = sess.graph.get_tensor_by_name("mixed_10/join:0")
        # reshape the tensor to shape (64, 2048):
        img_features_tensor = tf.reshape(img_features_tensor, (64, 2048))

        # apply the img transorm (get a tensor of shape (64, 300)):
        linear_transform = tf.matmul(img_features_tensor, W_img) + b_img
        img_features_tensor = tf.nn.sigmoid(linear_transform)

        for step, img_path in enumerate(img_paths):
            if step % 10 == 0:
                print step
                log(str(step))

            # read the image:
            img_data = gfile.FastGFile(img_path, "rb").read()
            try:
                # get the img features (np array of shape (64, 300)):
                img_features = sess.run(img_features_tensor,
                        feed_dict={"DecodeJpeg/contents:0": img_data})
                #img_features = np.float16(img_features)
            except:
                print "JPEG error for:"
                print img_path
                print "******************"
                log("JPEG error for:")
                log(img_path)
                log("******************")
            else:
                if not demo:
                    # get the image id:
                    img_name = img_path.split("/")[3]
                    img_id = img_name.split("_")[2].split(".")[0].lstrip("0")
                    img_id = int(img_id)
                else: # (if demo:)
                    # we're only extracting features for one img, (arbitrarily)
                    # set the img id to -1:
                    img_id = -1

                # save the img features to disk:
                cPickle.dump(img_features,
                        open("coco/data/img_features_attention/%d" % img_id, "wb"))

##########################################################
# Call based on if attention is needed or not
##########################################################
if(MODE == 1):
    # create a list of the paths to all imgs:
    img_paths = val_img_paths + test_img_paths + train_img_paths

    # extract all features:
    extract_img_features_attention(img_paths)
else:
   # get the feature vectors for all val imgs:
    val_img_id_2_feature_vector = extract_img_features(val_img_paths)
    # save on disk:
    cPickle.dump(val_img_id_2_feature_vector,
                 open("coco/data/val_img_id_2_feature_vector", "wb"))
    print "val done!"
    log("val done!")
    
    test_img_id_2_feature_vector = extract_img_features(test_img_paths)
    # save on disk:
    cPickle.dump(test_img_id_2_feature_vector,
                 open("coco/data/test_img_id_2_feature_vector", "wb"))
    print "test done!"
    log("test done!")
        # get the feature vectors for all train imgs:
    train_img_id_2_feature_vector = extract_img_features(train_img_paths)
    # save on disk:
    cPickle.dump(train_img_id_2_feature_vector,
                 open("coco/data/train_img_id_2_feature_vector", "wb"))


