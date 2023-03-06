# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:37:25 2023

@author: Luu Van Viet
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import fpdf
from PIL import Image

# Read file csv
data = pd.read_csv("SAT_GPA.csv")
# Show the description of data
data.describe()
# Set to training data (x, y)
y = data['GPA']
x = data['SAT']

#Prepare pdf file
pdf = fpdf.FPDF(format='letter')
pdf.add_page()
pdf.set_font("Arial", size=12)

# Remind that we need to put component x_0 = 1 to x
plt.scatter(x,y)
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show() 

#Prepare train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 60, shuffle = False)
Xbar = np.c_[ np.ones(60), X_train]

#Training model
regr = LinearRegression(fit_intercept= False)
regr.fit(Xbar, y_train)
pdf.write(5, "Model: SAT = {0:0.5f} + {1:0.5f}*GPA".format(regr.coef_[0], regr.coef_[1]))
pdf.ln()
#Draw regression line
plt.scatter(x,y)
yhat = regr.coef_[1]*x + regr.coef_[0]
fig = plt.plot(x,yhat, lw=4, c='orange', label = 'regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()
pdf.write(5, "Draw Regression Line:")
pdf.ln()
pdf.image('plot1.png',w=80,h=60)
pdf.ln()
#Calculate sum_square_error
X_test = np.array(X_test).reshape(-1,1)
X_test = np.c_[ np.ones(len(X_test)), X_test]
y_pred = regr.predict(X_test)
pdf.write(5,'Sum_square_error = {}'.format(mean_squared_error(y_test, y_pred)))

pdf.output('Luuvanviet_bai3_linearregression_pdf.pdf')