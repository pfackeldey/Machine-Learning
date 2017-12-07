#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import json
import os

# load acc and loss
loss = np.load('loss.npy')
val_loss = np.load('val_loss.npy')
acc = np.load('acc.npy')
val_acc = np.load('val_acc.npy'])

# plot loss
f = plt.figure()
plt.plot(loss)
plt.plot("val_loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["training loss", "validation loss"], loc="best")
f.savefig("loss.png")

# plot accuracy
f = plt.figure()
plt.plot(acc)
plt.plot(val_acc)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(["training accuracy", "validation accuracy"], loc="best")
f.savefig("accuracy.png")
