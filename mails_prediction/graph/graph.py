# coding: utf8

import networkx as nx
import pandas as pd
import community
import pickle


# import train data
train_set_sid = pd.read_csv('../data/training_set_sid.csv', names=['sender', 'id'])
train_info_sid = pd.read_csv('../data/training_info_sid.csv',
                             names=['id', 'datetime', 'content', 'recipients'])

# define a function to split a pd DataFrame column into multiple rows
def split_column(df, column, split_type='int', replace=True):
    """
    Args:
        df: a pd.DataFrame.
        column: a string.
        split_type: a string.
        replace: a boolean.
    Return:
        The same dataframe with column in parameters split into multiple rows,
        if replace=True the former column is replaced by the new one.
    """
    if split_type == 'int':
        split_col = df[column].str.split(' ').apply(pd.Series, 1).stack().astype(int)
    else:
        split_col = df[column].str.split(' ').apply(pd.Series, 1).stack()
    # to line up with df's index
    split_col.index = split_col.index.droplevel(-1)
    # needs a name to join
    split_col.name = 'split_' + column
    if replace:
        del df[column]
        split_col.name = column
    return df.join(split_col).reset_index(drop=True)


# split senders and merge with train info
train_set_sid = split_column(train_set_sid, 'id')
train = pd.merge(train_info_sid, train_set_sid, how='outer', on='id')

# split recipients and get pairwise mails count
train = split_column(train, 'recipients', split_type='str', replace=False)
mail_relations = train.groupby(['sender', 'split_recipients'])['id'].count()

# compute edges weights list with pairwise mails count
edges_weights = [[ix[0], ix[1], mail_nb] for ix, mail_nb in mail_relations.iteritems()]

# initialize a graph, add edges
graph = nx.Graph()
graph.add_weighted_edges_from(edges_weights)

# preform Louvain method for community detection
louvain_clusters = community.best_partition(graph)

# get the clustering coefficient for nodes
clustering_nodes_coef = nx.clustering(graph)

# get k-clique communities
kclique_clusters = {}
for i in range(2, nx.graph_clique_number(graph) + 1):
    clique = list(nx.community.k_clique_communities(graph, i))
    kclique_clusters[i] = list(clique[0])

# get Kernighan bisection communities
kernighan_bisection = nx.community.kernighan_lin_bisection(graph)

# get communities using the Girvan–Newman method.
comp = nx.community.girvan_newman(graph)
girvan_newman_clusters = tuple(sorted(c) for c in next(comp))

# get square clusters
square_clusters = nx.square_clustering(graph)

# get triangles number per node
nodes_triangles_nb = nx.triangles(graph)

# get nodes pagerank
nodes_pagerank = nx.pagerank(graph)

# save graph informations in pickle files
pickle.dump(girvan_newman_clusters, open('girvan_newman_clusters', 'wb'))
pickle.dump(clustering_nodes_coef, open('clustering_nodes_coef', 'wb'))
pickle.dump(kernighan_bisection, open('kernighan_bisection', 'wb'))
pickle.dump(nodes_triangles_nb, open('nodes_triangles_nb', 'wb'))
pickle.dump(louvain_clusters, open('louvain_clusters', 'wb'))
pickle.dump(kclique_clusters, open('kclique_clusters', 'wb'))
pickle.dump(square_clusters, open('square_clusters', 'wb'))
pickle.dump(nodes_pagerank, open('nodes_pagerank', 'wb'))
