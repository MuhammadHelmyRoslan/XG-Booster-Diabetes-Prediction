#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install xgboost


# In[3]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[4]:


# Load dataset
df = pd.read_csv("diabetes_dataset.csv")


# In[5]:


# Check initial data
print(df.head())
print(df.dtypes)
print(df.info())
print(df.shape)
print(df.describe())


# In[9]:


# Data preprocessing steps
# - Recoding smoking_history into categorical values
# - Categorizing gender and diabetes as categorical features
# - Adjusting the data types of the categorical and numerical columns accordingly
df['smoking_history'] = df['smoking_history'].replace(['former', 'ever', 'not current'], 'Non_active_Smoker')
df['smoking_history'] = df['smoking_history'].replace(['current'], 'Active_Smoker')
df['smoking_history'] = df['smoking_history'].replace(['never'], 'Non_Smoker')
CatCols = ['gender', 'smoking_history', 'diabetes']
NumCols = list(set(df.columns) - set(CatCols))
df[CatCols] = df[CatCols].apply(lambda x: x.astype('category'))
df[NumCols] = df[NumCols].apply(lambda x: x.astype('float64'))


# In[8]:


# Exploring categorical columns
for col in CatCols:
    if df[col].dtype.name == 'category':
        sns.countplot(x=col, data=df)
        plt.show()
        print(f"Unique values in {col}: {df[col].cat.categories}")
    else:
        print(f"Column {col} is not categorical.")


# In[11]:


# Ensure that we only include numeric columns for the correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, linewidths=2)
plt.show()


# In[14]:


# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)  # One-hot encoding

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42, stratify=y)

# Scale the entire dataset (now that it's all numeric)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[15]:


# PCA analysis
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# In[16]:


# Model training with XGBoost
xgbc = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, gamma=0, subsample=0.5, colsample_bytree=1, max_depth=8)
xgbc.fit(X_train_pca, y_train)


# In[17]:


# Predictions
predictions = xgbc.predict(X_test_pca)


# In[18]:


# Model Evaluation
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print('Accuracy Score:', round(accuracy_score(y_test, predictions), 2))



# In[19]:


# Comment on results
# - The XGBoost model exhibits high accuracy on the test data, but consider cross-validation and additional metrics for robust evaluation.


# In[ ]:




