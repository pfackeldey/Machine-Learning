# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

__all__ = [
    "KerasModels"
]


class KerasModels:

    # Class for a model using Keras backend Tensorflow

    def __init__(self, n_features, n_classes, learning_rate):

        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate

    def example_model(self):
        """
        3 (linear connected) layer example model:
        - 1 Dense (128) layer with dimension: n_features (training variables)
        - 1 Dropout layer (reduces overtraining effects)
        - 1 Dense (64) layer with dimension: 128
        - 1 Dropout layer (reduces overtraining effects)
        - 1 Dense layer with dimension: n_classes
        """

        model = Sequential()
        model.add(Dense(64, init='glorot_normal',
                        activation='relu', input_dim=self.n_features))
        model.add(Dropout(0.1))
        model.add(Dense(16, activation='relu', input_dim=64))
        model.add(Dropout(0.1))
        model.add(Dense(self.n_classes, activation='softmax'))

        # Compile the model:

        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            lr=self.learning_rate), metrics=['accuracy'])

        model.summary()
        model.save("example_model.h5")
