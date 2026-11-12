import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

data = {
    "Age":[25,31,np.nan,28,45,25],
    "City":["Mumbai","Pune","Mumbai","Nashik","Pune","Mumbai"],
    "Gender":["Male","Female","Male","Female","Male","Male"],
    "Income":[20000,25000,30000,np.nan,40000,45000]
}

df=pd.DataFrame(data)
print(df)

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Income"] = df["Income"].fillna(df["Income"].mean())
print(df)

df=df.drop_duplicates()
print(df)

df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
df = pd.get_dummies(df, columns=["City"], dtype=int)
print(df)

df[["Age","Income"]] = MinMaxScaler().fit_transform(df[["Age","Income"]])

print(df)