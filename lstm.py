import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import RandomSearch
from kerastuner import Objective
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

