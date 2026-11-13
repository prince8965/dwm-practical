# Experiment No 9 : Implementation of Market Basket Analysis Using Association Rule

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori , association_rules

# load dataset
data = pd.read_csv('Market_Basket.csv')
print('\nOriginal Datasets: \n' , data)

#Create Transaction
transaction = []

for i,row in data.drop(columns=['Transaction']).iterrows():
    transaction.append(row.dropna().astype(str).tolist())

print('\nSample ofTransaction: \n')
for t in transaction:
    print(t)

# convert transaction into boolean data
te = TransactionEncoder()
encoded = te.fit_transform(transaction)
df = pd.DataFrame(encoded , columns=te.columns_)

# Frequency Items
freq = apriori(df , min_support=0.05 , use_colnames=True )

print("\nFrequency Itemsets: \n" , freq)


# Generate Association rule
rules = association_rules(freq , metric='confidence' , min_threshold=0.30)

# select required columns
result = rules[['antecedents' , 'consequents', 'support' , 'confidence' , 'lift']]

result = result.sort_values('lift' , ascending=False)
strong = result[(result['confidence']>= 0.5) & (result['lift'] > 1)]

# Convert into string
strong = strong.copy()

strong['antecedents'] = strong['antecedents'].apply(lambda x : ','.join(sorted(x)))
strong['consequents'] = strong['consequents'].apply(lambda x : ','.join(sorted(x)))

# Final Display Output
print('\nFinal Association rule: ')
print(strong[['antecedents' , 'consequents' , 'support' , 'lift']])