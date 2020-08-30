import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
ed=pd.read_csv('D:\WORK_PYTHON\SIMPLE LINEAR REGRESSION\emp_data.csv')
## Simple Linear Regression ##
#Both x and y are contionous
#x-sh,y-cr
# EDA
ed.rename(columns = {'Salary_hike':'sh','Churn_outrate':'cr'},inplace=True)
ed.describe()
# Scatter Plot = Curvilinear in nature
plt.hist(ed.sh)
plt.hist(ed.cr)
plt.boxplot(ed.sh,0,"rs",0)
plt.boxplot(ed.cr)
plt.plot(ed.sh,ed.cr,"ro");plt.xlabel("sh");plt.ylabel("cr")
# Correlation Coefficient
ed.cr.corr(ed.sh)# cor()> 0.85 and negative direction
# Model Bulding # MR^2 > 0.81, P-value < 0.05 so it is significance
import statsmodels.formula.api as smf
model=smf.ols('cr~sh',data=ed).fit()
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(ed) # Predicted values of AT using the model
pred
import matplotlib.pylab as plt
plt.scatter(x=ed['sh'],y=ed['cr'],color='red');plt.plot(ed['sh'],pred,color='black');plt.xlabel('sh');plt.ylabel('cr')
