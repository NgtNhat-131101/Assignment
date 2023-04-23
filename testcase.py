from activation_np import *
from util import *
import numpy as np
import random
from dnn_np import *

np.random.seed(42)
n = random.randint(1, 12000)
train_x, train_y, test_x, test_y = get_bat_data()

print("X train shape is: " + str(train_x.shape))
print("y train shape is: " + str(train_y.shape))

# visualize_point(train_x, train_y, train_y)
print("\n===================================================================================")
# print("TEST CASE FOR TODO 1 and TODO 2")
layer = Layer((60, 100), "sigmoid")
unit_test_layer(layer)

print("\n===================================================================================")

