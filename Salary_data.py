import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
sd=pd.read_csv('D:\WORK_PYTHON\SIMPLE LINEAR REGRESSION\Salary_Data.csv')
## Simple Linear Regression ##
#Both x and y are contionous
#x-ye,y-s
# EDA
sd.rename(columns = {'YearsExperience':'ye','Salary':'s'},inplace=True)

sd.describe()
# Scatter Plot = Curvilinear in nature
plt.hist(sd.ye)
plt.hist(sd.s)
plt.boxplot(sd.ye,0,"rs",0)
plt.boxplot(sd.s)
plt.plot(sd.ye,sd.s,"ro");plt.xlabel("ye");plt.ylabel("s")
sd.s.corr(sd.ye)# cor()> 0.85 and positive direction
# Model Bulding # MR^2 > 0.81, P-value < 0.05 so it is significance
import statsmodels.formula.api as smf
model=smf.ols('s~ye',data=sd).fit()
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(sd) # Predicted values of AT using the model
pred
import matplotlib.pylab as plt
plt.scatter(x=sd['ye'],y=sd['s'],color='red');plt.plot(sd['ye'],pred,color='black');plt.xlabel('ye');plt.ylabel('s')
