# experiment no 10: Implementation of Data Mining and PageRank Algorithm with HITS Algorithm

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# create directed graph obj
G = nx.DiGraph()

# add edges
G.add_edges_from([
    ('A' , 'B'),
    ('A' , 'C'),
    ('B' , 'C'),
    ('C' , 'A'),
    ('D' , 'C'),
    ('D' , 'A'),
])

#draw the graph
nx.draw(G , with_labels=True, node_color='yellow' , node_size=2000 , arrows=True)
plt.title('Representation of Directed Graph')
plt.show()

# pagerank algorithm
pr = nx.pagerank(G)

print('PageTank Scores:')
for node , score in pr.items():
    print(node , ':' , round(score , 4))


# hits Algorithm
hub , authority = nx.hits(G)

print('\nHub Scores: ')
for node , score in hub.items():
    print(node , ':' , round(score , 4))


print('\nAuthority Scores: ')
for node , score in authority.items():
    print(node , ':' , round(score , 4))


# pagerank graph
plt.bar(pr.keys() , pr.values())
plt.title('PageRank Algorithm Graph')
plt.xlabel('Web Pages')
plt.ylabel('Rank')
plt.show()

# hits algorithm graph
x = np.arange(len(hub))

plt.bar(x - 0.2 , hub.values() , 0.4 , label='Hub')
plt.bar(x + 0.2 , authority.values() , 0.4 , label='Authority')

plt.xticks(x , hub.keys())
plt.title('HITS Algorithm Graph')
plt.xlabel('Web Pages')
plt.ylabel('Scores')
plt.legend()
plt.show()
