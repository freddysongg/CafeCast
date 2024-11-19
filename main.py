import warnings
warnings.filterwarnings("ignore", "urllib3 v2 only supports OpenSSL")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch

data = pd.read_excel('data/cafecast_data.xlsx')

print(data.head())
