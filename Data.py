import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv('D:/PTHB/Titanic/train.csv')
test = pd.read_csv('D:/PTHB/Titanic/test.csv')
combine = [train, test]

# Wrangle data
train = train.drop(['Ticket', 'Cabin'], axis = 1)
test = test.drop(['Ticket', 'Cabin'], axis = 1)
combine = [train, test]

# Creating new feature extracting from existing
#   - Retain the new Title feature for model training (title = Mr, Mrs, Miss, ...)
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# We can replace many titles with a more common name or classify them as `Others`.
def group_title(title):
    if title in ['Mr', 'Mrs', 'Miss', 'Master']:
        return title
    elif title == 'Ms':
        return 'Miss'
    else: 
        return 'Others'
train['Title'] = train['Title'].apply(lambda title: group_title(title))


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Others": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()


