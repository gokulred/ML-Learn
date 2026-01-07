import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


df = pd.read_csv('teams.csv')

# Split the data
df = df [["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

# Correlation 
"""print(teams.corr(numeric_only=True)["medals"])
sns.lmplot(x='athletes', y='medals', data=teams, fit_reg=True, ci=None)
plt.show()"""

# Check null values in the data 

df = df.dropna() 

# Split data 
train = teams[teams["year"]<2012].copy()
test = teams[teams["year"]>=2012].copy()

# Model definition
lr = LinearRegression()

# Predictors 
predictors = ['athletes', 'prev_medals']

# Target prediction 
target = 'medals'

# training the model 
lr.fit(train[predictors], train['medals'])

# test predictions 
predictions = lr.predict(test[predictors])

# print(predictions)

test['predictions'] = predictions

test.loc[test['predictions'] < 0, 'predictions'] = 0 # replacing values less than 0 with 0 

test['predictions'] = test['predictions'].round() # round values to nearest whole number 

# print(test)

# Mean absolute error calculation 

error = mean_absolute_error(test['medals'], test['predictions'])

# print(error)

# print(teams.describe()['medals'])  # check values: error should be less than std (standard deviation)

errors = (test['medals'] - test['predictions']).abs()

error_by_teams = errors.groupby(test['teams']).mean()

# print(error_by_teams) # error calculated based on teams 

