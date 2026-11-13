# Experiment no 5: Implementation of Data Visualizations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create dataset
data = {
    'Date':['2026-07-05' , '2026-07-10' , '2026-08-15' , '2026-08-20' , '2026-09-05' , '2026-09-15'],
    'Product' : ['Laptop' , 'Mobile' ,'Laptop' , 'Chair' , 'Mobile' , 'Chair'],
    'Category' : ['Electronics' , 'Electronics' , 'Electronics' , 'Furiture' , 'Electronics' , 'Furniture'],
    'City' : ['Mumbai' , 'Pune' , 'Nashik' , 'Mumbai' , 'Pune' , 'Nashik'],
    'Sales':[6000 , 4000 , 5000 , 2500 , 4500 , 1500 ],
    'Quantity' : [1 , 3 , 4 , 5 , 2 , 6]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

print('DataSet : \n' , df)

print('\nDataframe Shape : \n' , df.shape)

print('\n', df.isnull().sum()) 
print('\n', df.describe()) 
print('\nDuplicated Values: ', df.duplicated().sum()) 



city_sale = df.groupby('City')['Sales'].sum()
category_sale = df.groupby('Category')['Sales'].sum()
date_sale = df.groupby('Date')['Sales'].sum()


print(city_sale)
print(category_sale)
print(date_sale)


# bar graph - City by Sales
city_sale.plot(kind='bar')
plt.title('City by Sales')
plt.xlabel('City')
plt.ylabel('Sales')
plt.xticks(rotation=0)
plt.show()

# line graph - date by sales
plt.figure(figsize=(10 , 6))
plt.plot(date_sale.index , date_sale.values , marker='o')
plt.title('Sales Trend')
plt.xlabel('Dates')
plt.ylabel('Sales')
plt.show()

# scatter plot - quantity vs Sales
plt.figure(figsize=(10 , 6))
plt.scatter(df['Quantity'] , df['Sales'])
plt.title('Quantity Vs Sales')
plt.xlabel('Quantity')
plt.ylabel('Sales')
plt.show()

# pie chart - Sales by Category
plt.figure(figsize=(6 , 6))
plt.pie(category_sale.values , labels=category_sale.index , autopct='%1.1f%%')
plt.title('Sales Distribution by Category') 
plt.show() 

# histogram - sales distribution
plt.figure(figsize=(7 , 6))
plt.hist(df['Sales'] , bins=5 , edgecolor='black')
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()