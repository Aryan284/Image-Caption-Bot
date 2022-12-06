import numpy as np
from os import listdir
from pickle import dump #pickle is a library used for compression and storage of data
from keras.applications.vgg16 import VGG16 #VGG16 is a heavily pre-trained model used for image classification
from keras_preprocessing.image import load_img #this loads an image in the PIL (Python Imaging Library) format
from keras.models import Model

#converting PIL data to an numerical matrix with VGG16's specifications
def img_to_array(img):
    x = np.asarray(img, dtype="float32")
    if len(x.shape) == 3:
        return x
    elif len(x.shape) == 2:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")
    return x

#adjusting image data to fit the VGG16 model
def preprocess_images(x):
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68] #VGG16's specifications
    std = None
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    return x

def extract_features(directory):
	# load the model
	model = VGG16()
	# re-structure the model, cutting off the top layers related to image classification which we don't need
	# instead, the model with return the image features which we can re-purpose for the sake of captioning
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# summarize the model. the trained weights have already been downloaded
	print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_images(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store features
		features[image_id] = feature
		print('>%s' % name)
	return features

# extract features from all images
directory = 'Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# compress and save
dump(features, open('features.pkl', 'wb'))