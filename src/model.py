import os
import tensorflow as tf

# Model for 5 base channels + 5 means

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class face_model(tf.keras.Model):
    
    def __init__(self, params):
        super(face_model, self).__init__()
        img_size = (params.image_size, params.image_size, 10)
        intermediate_size = (params.image_size, params.image_size, 12)
        input_size = (params.image_size, params.image_size, 3)
        self.conversion_layer = tf.keras.layers.Conv2D(10, (3,3), padding='same', activation='relu', input_shape=img_size)
        self.conversion_layer_2 = tf.keras.layers.Conv2D(7, (3,3), padding='same', activation='relu')
        self.conversion_layer_1 = tf.keras.layers.Conv2D(3, (1,1), padding='same', activation='relu', input_shape=input_size)
        self.base_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=input_size)
        self.base_model.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.embedding_layer = tf.keras.layers.Dense(units=params.embedding_size)

    def call(self, images):
        x = self.conversion_layer(images)
        x = self.conversion_layer_2(x)
        x = self.conversion_layer_1(x)
        x = self.base_model(x)
        x = self.flatten(x)
        x = self.embedding_layer(x)
        return x