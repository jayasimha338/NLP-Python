import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
sd=pd.read_csv("D:/WORK_PYTHON/MULTI LINEAR REGRESSION/50_Startups.csv")
#Multi-Linear Regression
#Where all variables x1,x2,x3,x4,x5,x6 and y are continous
#y-pt,x1-rd,x2-ad,x3-ms,x4-sc,x5-sf,x6-sn
# Exploratory Data Analysis
sd1=sd.replace('New York',0)
sd1=sd1.replace('California',1)
sd1=sd1.replace('Florida',2)
sd1.rename(columns = {'R&D Spend':'rd','Administration':'ad','Marketing Spend':'ms','State':'st','Profit':'pt'},inplace=True)
sd1.describe()
type(sd1)
sd1.head(10)
# Correlation matrix 
sd1.corr()
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(sd1.iloc[:,:])
# columns names
sd1.columns
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('pt~rd+ad+ms+st',data=sd1).fit() # regression model

# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary()

pred = ml1.predict(sd1) # Predicted values of profit using the model
pred
# p-values for ad,ms,st are more than 0.05 
# preparing model based only on ad
ml_ad=smf.ols('pt~ad',data = sd1).fit()  
ml_ad.summary() 
# p-value !<0.05 .. It is  no significant 

# Preparing model based only on ms
ml_ms=smf.ols('pt~ms',data = sd1).fit()  
ml_ms.summary() 
## p-value <0.05 ..It is significant 
# Preparing model based only on st
ml_st=smf.ols('pt~st',data = sd1).fit()  
ml_st.summary() 
# p-value !<0.05 .. It is  no significant 
# Preparing model based only on WT ad,ms,st
ml_ams=smf.ols('pt~ad+ms+st',data = sd1).fit()  
ml_ams.summary() 
# Where ad,ms are significant but st is not
# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# index 45,46,48,49 is showing high influence so we can exclude that entire row
sd1_new=sd1.drop(sd1.index[[45,46,48,49]],axis=0)
sd1_new
# Preparing model                  
ml_new = smf.ols('pt~rd+ad+ms+st',data = sd1_new).fit()    

# Getting coefficients of variables        
ml_new.params

# Summary
ml_new.summary() 

## Now applying transformation

## Exponential Model # Here p-value is not significant, MR^2 is moderate = 0.762
model3 = smf.ols('np.log(pt)~rd+ad+ms+st',data=sd1).fit()
model3.params
model3.summary()


#Logrithamic Model # Here p-value is significant, MR^2 is moderate = 0.669
model2 = smf.ols('pt~np.log(rd+ad+ms+st)',data=sd1).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(sd1)
pred2

# added variable plot for the final model
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model2)

## From above Logrithamic Model here p-value is significant, MR^2 is moderate = 0.669