# Experiment no 7 - Implementation of Customer Segmentation using Clustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# load dataset
df = pd.read_csv('Mall_Customers.csv')
print('\nDataset: \n' , df.head())

print('\nMissing Values: \n' , df.isnull().sum())

print('\nDuplicated Values : ' , df.duplicated().sum())

# we want only annual income and spending amount 
x = df[['Annual Income (k$)' , 'Spending Score (1-100)']]

# calculate wcss
wcss = []

for k in range(1 , 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit_predict(x)
    wcss.append(kmeans.inertia_)

# plotting the elbow method on graph
plt.figure(figsize=(10 , 7))
plt.plot(
    range(1 , 11),
    wcss , 
    marker='o'
)
plt.title('Elbow Method')
plt.xlabel('No of Clusters')
plt.ylabel('WCSS')
plt.show()

# now apply clustering 
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(x)

print('\nDividing Dataset into Clusters: \n' , df.sample(7))

# calculate centroid
centroid = kmeans.cluster_centers_
print('\nCentroids : \n' , centroid)


# plotting Clusters and its centroid on the scatter graph

plt.figure(figsize=(10 , 10))
plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    s=60
)

plt.scatter(
    centroid[: , 0],
    centroid[: , 1],
    color='red',
    marker='X',
    s=250
)

plt.title('Customer Segmentations using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()