import tensorflow as tf
import os
#import keras 
from keras import optimizers
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
import numpy as np
from keras.models import Model 
from keras.layers import Input , Flatten , Dense
from keras import backend as K


input_train01 = 'DATA/Fake/Train/Input01'
input_train02 = 'DATA/Fake/Train/Input02'
output_train = 'DATA/Fake/Train/Output'

input_test01 = 'DATA/Fake/Test/Input01'
input_test02 = 'DATA/Fake/Test/Input02'
output_test = 'DATA/Fake/Test/Output'

##Preparamos nuestras imagenes
datagen = image.ImageDataGenerator(rescale=1. / 255)

input_train01 = datagen.flow_from_directory(input_train01,batch_size=32,class_mode='categorical')
input_train02 = datagen.flow_from_directory(input_train02,batch_size=32,class_mode='categorical')
output_train = datagen.flow_from_directory(output_train,batch_size=32,class_mode='categorical')

input_test01 = datagen.flow_from_directory(input_test01,batch_size=32,class_mode='categorical')
input_test02 = datagen.flow_from_directory(input_test02,batch_size=32,class_mode='categorical')
output_test = datagen.flow_from_directory(output_test,batch_size=32,class_mode='categorical')

#Function to retrieve features from intermediate layers
def get_activations(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
    activations = get_activations([X_batch,0])
    return activations

#Function to extract features from intermediate layers
def extra_feat(img_path):
        #Using a VGG19 as feature extractor
        base_model = VGG19(weights='imagenet',include_top=False)
        img = image.load_img(img_path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        block1_pool_features=get_activations(base_model, 3, x)
        block2_pool_features=get_activations(base_model, 6, x)
        block3_pool_features=get_activations(base_model, 10, x)
        block4_pool_features=get_activations(base_model, 14, x)
        block5_pool_features=get_activations(base_model, 18, x)
    
        x1 = tf.image.resize_images(block1_pool_features[0],[112,112])
        x2 = tf.image.resize_images(block2_pool_features[0],[112,112])
        x3 = tf.image.resize_images(block3_pool_features[0],[112,112])
        x4 = tf.image.resize_images(block4_pool_features[0],[112,112])
        x5 = tf.image.resize_images(block5_pool_features[0],[112,112])
        
        F = tf.concat([x1,x2,x3,x4,x5],3) #Change to only x1, x1+x2,x1+x2+x3..so on, in order to visualize features from different blocks
        return F

def main():

  sess = tf.InteractiveSession()
  input_img1 = Input((224,224,3))
  input_img2 = Input((224,224,3))
  
  process_img1 = extra_feat(input_img1)
  process_img2 = extra_feat(input_img2)
  
  pair = np.append(process_img1,process_img2,axis=0)
  
  cnn=Flatten()(pair)
  cnn=Dense(15000 , activation='relu')(cnn)
  prediction=Dense(12544 , activation='softmax')(cnn)

  # creating the final model 
  VL2_model = Model(input = [input_img1 , input_img2], output = prediction)
  
  VL2_model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.0005),metrics=['accuracy'])
  VL2_model.fit(x=[input_train01,input_train02],y=output_train,steps_per_epoch=1000,epochs=20,validation_data=[[input_test01,input_test02],output_test],validation_steps=300)
  
  target_dir = 'modelo/'
  if not os.path.exists(target_dir):
      os.mkdir(target_dir)
      VL2_model.save('modelo/modelo.h5')
      VL2_model.save_weights('modelo/pesos.h5')


if __name__ == "__main__":
    main()
