import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
tc=pd.read_csv("D://WORK_PYTHON//MULTI LINEAR REGRESSION//ToyotaCorolla.csv",encoding='latin1')
#You have to use the encoding as latin1 to read this file as there are some special character in this file
tc1 = []
tc1=tc[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
#Multi-Linear Regression
#Where all variables x1,x2,x3,x4,x5,x6,x7,x8 and y are continous
#y- predicting model price,x1,x2,x3,x4,x5,x6,x7,x8 are independent variables
# Exploratory Data Analysis
tc1.describe()
# Correlation matrix 
tc1.corr()
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(tc1.iloc[:,:])
# columns names
tc1.columns
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=tc1).fit() # regression model

# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary()

pred = ml1.predict(tc1) # Predicted values of price using the model
pred
# p-values for cc,Doors are more than 0.05 
# preparing model based only on cc
ml_cc=smf.ols('Price~cc',data = tc1).fit()  
ml_cc.summary() 
## p-value <0.05 ..It is significant 
# preparing model based only on ad
ml_doors=smf.ols('Price~Doors',data = tc1).fit()  
ml_doors.summary() 
## p-value <0.05 ..It is significant 
# preparing model on both cc and doors
ml_cd=smf.ols('Price~cc+Doors',data = tc1).fit()  
ml_cd.summary() 
## p-value <0.05 ..It is significant for both
# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# index 80,221,960 is showing high influence so we can exclude that entire row
tc1_new=tc1.drop(tc1.index[[80,221,960]],axis=0)
tc1_new
# Preparing model                  
ml_new = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data = tc1_new).fit()    

# Getting coefficients of variables        
ml_new.params

# Summary
ml_new.summary() 
# added variable plot for the final model
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(ml_new)
## From above we can conclude the mlr1 model is significant where P-value < 0.05 ,MR^2 >0.81 
# Where we have influential values in 81th,222th,961th observations are affecting the model

