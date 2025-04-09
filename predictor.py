import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# load dataframe
dataframe = pd.read_csv('./dataset.csv', delimiter=',')
X = dataframe.iloc[:, 0:8]
y = dataframe.iloc[:, 8].values

# switch categorical variables for numeric values
encoder_set = dataframe.iloc[:, 5:7].values
oe = OrdinalEncoder(dtype=int)
encoded_data = oe.fit_transform(encoder_set)

# drop old values and replace them by numeric values
final_dataframe = X.drop(columns=['action_name', 'algorithm_name'])
final_dataframe.insert(5, 'action_name', encoded_data[:, 0])
final_dataframe.insert(6, 'algorithm_name', encoded_data[:, 1])

# test
X_train, X_test, y_train, y_test = train_test_split(
    final_dataframe.values, y, test_size=0.2, random_state=0)

# random forest alg
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
