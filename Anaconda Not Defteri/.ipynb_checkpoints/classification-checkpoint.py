# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)

dFrame = pd.read_excel('/Users/furkanyamaner/Desktop/Anaconda Not Defteri/Data_processed.xlsx')
nullValue = 0.000000

columns = dFrame.columns
for col in dFrame.columns:
    
    if nullValue in dFrame[col].values:
        dFrame[col] = dFrame[col].mean()

"""print(dFrame["District_Ara"].head(100))"""
ncols = len(dFrame.columns)
numeric_cols = dFrame.select_dtypes(include=["number"])

dFrame = dFrame.dropna()
dFrame.to_excel("temizlenmis_data.xlsx", index="false")
"""for i in range(ncols):
    for j in range(ncols):
        plt.scatter(dFrame.index[i],dFrame.index[j])
plt.show()"""

new_dFrame = pd.read_excel('/Users/furkanyamaner/Desktop/DERSLER/464Proje/temizlenmis_data.xlsx')
print(new_dFrame.head(10))