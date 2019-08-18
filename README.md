# Financial-Inclusion
##### Short Desc
Financial inclusion is where individuals and businesses have access to useful and affordable financial products and services that meet their needs that are delivered in a responsible and sustainable way.
**The purpose for this project**

### The research problem is to figure out how we can predict which individuals are most likely to have or use a bank account. Your solution will help provide an indication of the state of financial inclusion in Kenya, Rwanda, Tanzania, and Uganda, while providing insights into some of the key demographic factors that might drive individuals’ financial outcomes.

**Libraries and Imports**

```
# All imports

# Importing Numpy
import numpy as np

# Importing Pandas
import pandas as pd

# Importing Matplotlib
import matplotlib.pyplot as plt


# Importing Seaborn
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# standazing
from sklearn.preprocessing import StandardScaler
# PCA
from sklearn.decomposition import PCA
making predictions
from sklearn.ensemble import RandomForestClassifier

# Performance Evaluation
# This is by using confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# Accuracy
from sklearn.metrics import accuracy_score
```
![Image of Yaktocat](https://github.com/codybaraks/Financial-Inclusion/blob/master/household%20size.PNG)
#### A Graph of Household-Size in comparison to the Age of respondent

