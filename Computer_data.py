import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
cd=pd.read_csv("D:/WORK_PYTHON/MULTI LINEAR REGRESSION/Computer_Data.csv")
#Multi-Linear Regression
#Where all variables x1,x2,x3,x4,x5,x6,x7,x8,x9 and y are continous
#y-price of computer,x1,x2,x3,x4,x5,x6,x7,x8,x9 are independent variables
# Exploratory Data Analysis
cd=cd.drop('Unnamed: 0',axis=1)
cd=cd.replace('yes',1)
cd=cd.replace('no',0)
cd.describe()
# Correlation matrix 
cd.corr()
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cd.iloc[:,:])
# columns names
cd.columns
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=cd).fit() # regression model

# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary()

pred = ml1.predict(cd) # Predicted values of price using the model
pred
# added variable plot for the final model
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(ml1)
## From above we can conclude the mlr1 model is significant where P-value < 0.05 ,MR^2 >0.81 