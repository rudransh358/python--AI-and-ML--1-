import numpy as np
import pandas as pd
from  sklearn.linear_model import LinearRegression
import math
def predict_using_sklearn():
    df=pd.read_csv('test_scores.csv')
    r=LinearRegression()
    r.fit([['math']],df.cs)
    return r.coef_,r.intercept_
def gradient_descent(x,y):
    m_curr=0
    b_curr=0
    learning_rate=0
    n=len(x)
    iteration=0

    cost_prediction=0
    
    for i in range(iteration):
        y_predicted=m_curr*x + b_curr
        cost=(1/n)*sum(y-y_predicted)^2
        md=(2/n)*sum-x(y-y_predicted)
        bd=(2/n)*sum-(y-y_predicted)
        m_curr=m_curr-learning_rate*md
        b_curr=b_curr-learning_rate*bd
        if math.isclose('cost',cost_prediction,rel_t0l=1e-20):
            break
        cost=cost_prediction
        print("m{},b{},cost{},i{}".format(m_curr,b_curr,cost,i))
        return m_curr,b_curr
