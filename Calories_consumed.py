# For reading data set
# importing necessary libraries
import pandas as pd 
import matplotlib.pyplot as plt
cc=pd.read_csv('D:\WORK_PYTHON\SIMPLE LINEAR REGRESSION\calories_consumed.csv')
## Simple Linear Regression ##
#Both x and y are contionous
#x-calories,y-weight
# EDA
cc.rename(columns = {'Calories Consumed':'cal'},inplace=True)
cc.describe()
# Scatter Plot = Curvilinear in nature
plt.hist(cc.wg)
plt.hist(cc.cal)
plt.boxplot(cc.wg,0,"rs",0)
plt.boxplot(cc.cal)
plt.plot(cc.wg,cc.cal,"ro");plt.xlabel("wg");plt.ylabel("cal")
# Correlation Coefficient
cc.cal.corr(cc.wg)# cor()> 0.85 and positive direction
# Model Bulding # MR^2 > 0.81, P-value < 0.05 so it is significance
import statsmodels.formula.api as smf
model=smf.ols("cal~wg",data=cc).fit()
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(cc) # Predicted values of AT using the model
pred
import matplotlib.pylab as plt
plt.scatter(x=cc['wg'],y=cc['cal'],color='red');plt.plot(cc['wg'],pred,color='black');plt.xlabel('wg');plt.ylabel('cal')
