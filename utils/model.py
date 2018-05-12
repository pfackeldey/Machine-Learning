# -*- coding: utf-8 -*-

import os
import yaml
import six

from keras.models import Sequential
from keras.layers import Dense, Dropout, AlphaDropout, BatchNormalization, Activation
from keras.optimizers import Adam, Nadam, SGD, RMSprop
from keras import regularizers


__all__ = [
    "Config", "KerasModel"
]


class Config(object):

    def __init__(self):
        path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../tasks/analysis/"))
        self.load = yaml.load(
            open(os.path.join(path, "MSSM_HWW.yaml"), "r"))


class Base(Config):

    def __init__(self, *args, **kwargs):
        super(Base, self).__init__(*args, **kwargs)

        self._nfeatures = len(self.load["features"])
        self._nclasses = len(self.load["classes"])
        self._learning_rate = 0.01

    @property
    def nfeatures(self):
        return self._nfeatures

    @property
    def nclasses(self):
        return self._nclasses

    @property
    def lr(self):
        return self._learning_rate

    @lr.setter
    def lr(self, value):
        self._learning_rate = value


class KerasModel(Base):

    # Class for a model using Keras backend Tensorflow

    def __init__(self, *args, **kwargs):
        super(KerasModel, self).__init__(*args, **kwargs)

        self._plot_model = False
        self._modelname = "model"

    @property
    def plot_model(self):
        return self._plot_model

    @plot_model.setter
    def plot_model(self, value):
        if not isinstance(value, bool):
            raise TypeError("plot_model must be set to a boolean")
        else:
            self._plot_model = value

    @property
    def modelname(self):
        return self._modelname

    @modelname.setter
    def modelname(self, value):
        if not isinstance(value, six.string_types):
            raise TypeError("modelname must be set to a string")
        else:
            self._modelname = self.__class__.__name__.lower() + "_fold" + value + ".h5"

    def example_model(self):
        """
        5 (linear connected) layer example model:
        - 1 Dense (128) layer with dimension: nfeatures (training variables)
        - 1 Dropout layer (reduces overtraining effects)
        - 1 Dense (64) layer with dimension: 128
        - 1 Dropout layer (reduces overtraining effects)
        - 1 Dense layer with dimension: nclasses
        """
        model = Sequential()
        model.add(Dense(128, kernel_initializer='glorot_normal',
                        activation='relu', input_dim=self.nfeatures))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu', input_dim=128))
        model.add(Dropout(0.1))
        model.add(Dense(self.nclasses, activation='softmax'))

        # Compile the model:
        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            lr=self.lr), metrics=['accuracy'])

        model.summary()
        model.save(self.modelname)

        if self.plot_model:
            # Visualize model as graph
            try:
                from keras.utils.visualize_util import plot
                plot(model, to_file='model.png', show_shapes=True)
            except:
                print('[INFO] Failed to make model plot')

        return model

    def binary_MSSM_HWW_model(self):
        """
        Binary classification model Signal vs. Background
        """
        model = Sequential()
        model.add(Dense(128, kernel_initializer='glorot_normal',
                        activation='relu', input_dim=self.nfeatures))
        model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu', input_dim=128))
        model.add(Dropout(0.1))
        model.add(Dense(self.nclasses, activation='softmax'))

        # Compile the model:
        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            lr=self.lr), metrics=['accuracy'])

        model.summary()
        model.save(self.modelname)

        if self.plot_model:
            # Visualize model as graph
            try:
                from keras.utils.visualize_util import plot
                plot(model, to_file='model.png', show_shapes=True)
            except:
                print('[INFO] Failed to make model plot')

        return model

    def multiclass_MSSM_HWW_model(self):
        """
        Multiclassification model
        """
        model = Sequential()
        model.add(Dense(64, input_dim=self.nfeatures))
        # model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.2))
        for i in range(3):
            model.add(Dense(64))
            model.add(Activation('elu'))
            model.add(BatchNormalization())
            # model.add(Dropout(0.2))
        model.add(Dense(self.nclasses, activation='softmax'))

        # Compile the model:
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.lr), metrics=['acc'])

        model.summary()
        """
        model.save(self.modelname)
        if self.plot_model:
            # Visualize model as graph
            try:
                from keras.utils.visualize_util import plot
                plot(model, to_file='model.png', show_shapes=True)
            except:
                print('[INFO] Failed to make model plot')
        """
        return model

    def multiclass_MSSM_HWW_testmodel(self):
        """
        Multiclassification model
        """
        model = Sequential()
        model.add(Dense(64, kernel_initializer='lecun_normal',
                        input_dim=self.nfeatures, activation='selu'))
        # model.add(BatchNormalization())
        for i in range(9):
                # model.add(AlphaDropout(0.2))
            model.add(
                Dense(64, kernel_initializer='lecun_normal', activation='selu'))
            # model.add(BatchNormalization())
        model.add(Dense(self.nclasses, activation='softmax'))

        # Compile the model:
        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            lr=self.lr), metrics=['accuracy'])

        model.summary()
        model.save(self.modelname)

        return model
