# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

__all__ = [
    "KerasModels"
]


class KerasModels:

    # Class for a model using Keras backend Tensorflow

    def __init__(self, n_features, n_classes, learning_rate, plot_model):

        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.plot_model = plot_model

    def example_model(self):
        """
        5 (linear connected) layer example model:
        - 1 Dense (128) layer with dimension: n_features (training variables)
        - 1 Dropout layer (reduces overtraining effects)
        - 1 Dense (64) layer with dimension: 128
        - 1 Dropout layer (reduces overtraining effects)
        - 1 Dense layer with dimension: n_classes
        """

        model = Sequential()
        model.add(Dense(128, init='glorot_normal',
                        activation='relu', input_dim=self.n_features))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu', input_dim=128))
        model.add(Dropout(0.1))
        model.add(Dense(self.n_classes, activation='softmax'))

        # Compile the model:

        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            lr=self.learning_rate), metrics=['accuracy'])

        model.summary()
        model.save("example_model.h5")

        if self.plot_model:
            # Visualize model as graph
            try:
                from keras.utils.visualize_util import plot
                plot(model, to_file='model.png', show_shapes=True)
            except:
                print('[INFO] Failed to make model plot')

    def binary_MSSM_HWW_model(self):
        """
        Binary classification model Signal vs. Background
        """
        model = Sequential()
        model.add(Dense(128, init='glorot_normal',
                        activation='relu', input_dim=self.n_features))
        model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu', input_dim=128))
        model.add(Dropout(0.1))
        model.add(Dense(self.n_classes, activation='softmax'))

        # Compile the model:

        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            lr=self.learning_rate), metrics=['accuracy'])

        model.summary()
        model.save("binary_MSSM_HWW_model.h5")

        if self.plot_model:
            # Visualize model as graph
            try:
                from keras.utils.visualize_util import plot
                plot(model, to_file='model.png', show_shapes=True)
            except:
                print('[INFO] Failed to make model plot')

    def multiclass_MSSM_HWW_model(self):
        """
        Multiclassification model
        """
        model = Sequential()
        model.add(Dense(256, init='glorot_normal',
                        activation='relu', input_dim=self.n_features))
        model.add(Dropout(0.1))
        model.add(Dense(256, activation='relu', input_dim=256))
        model.add(Dropout(0.1))
        model.add(Dense(256, activation='relu', input_dim=256))
        model.add(Dense(self.n_classes, activation='softmax'))

        # Compile the model:

        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            lr=self.learning_rate), metrics=['accuracy'])

        model.summary()
        model.save("multiclass_MSSM_HWW_model.h5")

        if self.plot_model:
            # Visualize model as graph
            try:
                from keras.utils.visualize_util import plot
                plot(model, to_file='model.png', show_shapes=True)
            except:
                print('[INFO] Failed to make model plot')
