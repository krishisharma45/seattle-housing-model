import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv('kc_house_data.csv')

X = data[['lat', 'waterfront', 'view', 'floors', 'bathrooms', 'sqft_living']]
y = data['price']
lr = LinearRegression()
model = lr.fit(X, y)
filename = 'king_model.sav'
pickle.dump(model, open(filename, 'wb'))
