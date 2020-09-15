import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

scale = StandardScaler()

path = 'C:/Users/julio/Documents/Datasets/insurance.csv'
df = pd.read_csv(path)

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

print(df.corrwith(df['charges']))

X = df.drop('charges',axis = 1)
y = df['charges']

X_scaled = scale.fit_transform(X)
X_training, X_testing, y_training, y_testing = train_test_split(X_scaled, y, test_size=0.2, random_state=1)
forest_model = RandomForestRegressor(n_estimators=1000, random_state=42)
forest_model.fit(X_training, y_training)

print(forest_model.score(X_testing, y_testing))

score = cross_val_score(forest_model, X_training,y_training, cv=10)
print(score)
print(score.mean())

rf_preds = forest_model.predict([[42,1,23.4,1,0,1]])
print(rf_preds)

print(forest_model.feature_importances_)
feat_importances = pd.Series(forest_model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.title('RandomForest Feature impotance')
plt.show()