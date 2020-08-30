import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
det=pd.read_csv('D:\WORK_PYTHON\SIMPLE LINEAR REGRESSION\delivery_time.csv')
## Simple Linear Regression ##
#Both x and y are contionous
#x-sorting time,y-delivery time
# EDA
det.rename(columns = {'Delivery Time':'dt','Sorting Time':'st'},inplace=True)
det.describe()
# Scatter Plot = Curvilinear in nature
plt.hist(det.dt)
plt.hist(det.st)
plt.boxplot(det.dt,0,"rs",0)
plt.boxplot(det.st)
plt.plot(det.dt,det.st,"ro");plt.xlabel("dt");plt.ylabel("st")
# Correlation Coefficient
det.st.corr(det.dt)# cor()= 0.825 and positive direction
# Model Bulding ## Here p-value is not significant
import statsmodels.formula.api as smf
model=smf.ols("st~dt",data=det).fit()
model.params
model.summary()
model.conf_int(0.05) # 95% confidence interval

pred = model.predict(det) # Predicted values of AT using the model
pred
import matplotlib.pylab as plt
plt.scatter(x=det['dt'],y=det['st'],color='red');plt.plot(det['dt'],pred,color='black');plt.xlabel('dt');plt.ylabel('st')

#Logrithamic Model
model2 = smf.ols('dt~np.log(st)',data=det).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(det)
pred2.corr(det.dt)
# pred2 = model2.predict(wcat.iloc[:,0])
pred2
plt.scatter(x=det['dt'],y=det['st'],color='red');plt.plot(det['dt'],pred2,color='black');plt.xlabel('dt');plt.ylabel('st')

## Exponential Model # Here p-value is significant, MR^2 is moderate = 0.711
model3 = smf.ols('np.log(dt)~st',data=det).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(det)
pred_log
pred3=np.exp(pred_log)  # as we have used log(AT) in preparing model so we need to convert it back
pred3
pred3.corr(det.dt)
plt.scatter(x=det['dt'],y=det['st'],color='green');plt.plot(det.st,np.exp(pred_log),color='blue');plt.xlabel('dt');plt.ylabel('dt')
resid_3 = pred3-det.dt

# Quadratic model
det["st_Sq"] = det.st*det.st
model_quad = smf.ols("dt~st+st_Sq",data=det).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(det)

model_quad.conf_int(0.05) # 
plt.scatter(det.dt,det.st,c="b");plt.plot(det.st,pred_quad,"r")

# 
# From above all model bulding we can conclude thate exponential model is some what significance
#where P-value < 0.05,MR^2 = 0.7109 is moderate
