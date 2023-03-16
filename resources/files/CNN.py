# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as k
import json

k.clear_session()


# Read parameters ###########################################################
with open("parameters.json") as data:
    parameters=json.loads(data.read())


# Initial conditions ########################################################
folder='resources/files/'
classes=parameters['classes']
iterations=parameters['iterations']*classes
quantity=parameters['quantity']
height, width=parameters['height'], parameters['width']

chanels=parameters['chanels']
neurons=parameters['neurons']

data_training=folder+'training'
data_validation=folder+'validation'
filtrosconv1=32
filtrosconv2=64
filtrosconv3=128
tam_filtro1=(4,4)
tam_filtro2=(3,3)
tam_filtro3=(2,2)
tam_pool=(2,2)

preprocesamiento_entre=ImageDataGenerator(
    rescale=1./255
)

preprocesamiento_vali=ImageDataGenerator(
    rescale=1./255
)

image_training=preprocesamiento_entre.flow_from_directory(
    data_training,
    target_size=(height, width),
    class_mode='categorical',
    color_mode='grayscale',
    batch_size=1
    
)

image_validation=preprocesamiento_vali.flow_from_directory(
    data_validation,
    target_size=(height, width),
    class_mode='categorical',
    color_mode='grayscale',
    batch_size=1
)

#Red neuronal convolucional (CNN)
cnn=Sequential()
cnn.add(Convolution2D(filtrosconv1, 
                        tam_filtro1, 
                        padding='same', 
                        input_shape=(height, width, chanels), 
                        activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Convolution2D(filtrosconv2, tam_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Convolution2D(filtrosconv3, tam_filtro3, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Flatten())
cnn.add(Dense(neurons, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(classes, activation='softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn.fit(image_training, 
        steps_per_epoch=quantity, 
        epochs=iterations, 
        validation_data=image_validation, 
        validation_steps=quantity)
cnn.save(folder+'model.h5')
cnn.save_weights(folder+'weights.h5')