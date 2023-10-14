# Ex-06 Feature Transformation
## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.


## EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
## ALGORITHM:
### STEP 1:
Read the given Data

### STEP 2:
Clean the Data Set using Data Cleaning Process

### STEP 3:
Apply Feature Transformation techniques to all the features of the data set

### STEP 4:
Print the transformed features
## PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer 
from sklearn.preprocessing import QuantileTransformer

df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```

## OUTPUT:
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/e9f87bcb-5049-429d-a4a5-d46402061e17)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/308144dd-090c-4074-b8a3-4476829f2b6b)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/d6842d49-8409-4a2f-a4a8-8a69af3e6826)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/77d933c2-8a5b-4adb-b454-5d91ad93939a)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/ddc6d7cf-3522-4a5f-81f8-035edb9182c4)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/d5b1595a-fef6-44b0-8738-18a6d7a8e5ef)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/fac9a53a-f60c-4fb5-a62f-169bedaf6944)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/13a8b31b-6254-437d-97f9-18db3f6dadbe)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/c89ae7e9-4a41-496d-94db-5e8a48fb771a)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/83fb611b-2ff0-4050-905d-efb04637c1a5)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/edd427eb-7410-4f79-bd81-35ac4e879885)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/758c8d13-24fa-44aa-ab1f-3941d9391382)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/1ad8f5cf-6f0e-4716-acd7-30bb91efdf13)
![image](https://github.com/Evangelin-Ruth/ODD2023-Datascience-Ex06/assets/94219798/8f32b981-6bed-4a84-96ce-2278fd2ebb25)


## RESULT:
Thus feature transformation is done for the given dataset.


