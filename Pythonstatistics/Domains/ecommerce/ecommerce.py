import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv('Trafficdata.csv')
print(df.head())  # View the first few rows
print(df.info())  # Check column types and null values
pd.set_option('display.max_columns',None)
print(df.describe())