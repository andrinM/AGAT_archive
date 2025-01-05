from sklearn import preprocessing
import numpy as np
import pandas as pd
import sys
import os
current_dir = os.getcwd()  
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Parent directory of import_destination (i.e., src)
sys.path.append(src_dir)
from Data import testData
area = [142,
147,
85,
175,
160,
201,
204,
187,
130,
250,]

price = np.array([232,
515,
178,
489,
510,
650,
548,
520,
380,
670])

total = (price - 469.2)**2

print(sum(total))
