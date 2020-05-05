###Import the nedded libraries

%tensorflow_version 1.x #specifies which version of tensrflow to use. 
                        #It can be chosen only between macroversions 1 and 2
                        #We chose version 1 for the frozen graph support
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import train, Session
import matplotlib.pyplot as plt

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil

import urllib.request
import tarfile
