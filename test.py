from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense
from keras.models import Model

# Define two input layers
image_input = Input((32, 32, 3))
vector_input = Input((6,))

# Convolution + Flatten for the image
conv_layer = Conv2D(32, (3,3))(image_input)
flat_layer = Flatten()(conv_layer)

# Concatenate the convolutional features and the vector input
concat_layer= Concatenate()([vector_input, flat_layer])
output = Dense(3)(concat_layer)

# define a model with a list of two inputs
model = Model(inputs=[image_input, vector_input], outputs=output)
