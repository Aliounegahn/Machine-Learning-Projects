# coding: utf8

import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier
from ml_metrics import mapk
from time import time


# import train data
train_set_sid = pd.read_csv('data/training_set_sid.csv', names=['sender', 'id'])
train_info_sid = pd.read_csv('data/training_info_sid.csv',
                             names=['id', 'datetime', 'content', 'recipients'])

# import test data
test_random_sid = pd.read_csv('data/test_random_sid.csv', names=['id', 'recipients'], skiprows=1)
test_info_sid = pd.read_csv('data/test_info_sid.csv', names=['id', 'datetime', 'content'])
test_set_sid = pd.read_csv('data/test_set_sid.csv', names=['sender', 'id'])

# concat train and test
df = pd.concat([train_info_sid, test_info_sid])

# define a function to split a pd df column into multiple rows
def split_column(df, column, replace=True):
    # split and stack away the column
    split_col = df[column].str.split(' ').apply(pd.Series, 1).stack().astype(int)
    # line up with df's index
    split_col.index = split_col.index.droplevel(-1)
    # needs a name to join
    split_col.name = 'split_' + column
    # replace if asked
    if replace:
        del df[column]
        split_col.name = column
    return df.join(split_col).reset_index(drop=True)


# split senders emails id into multiple rows
train_set_sid = split_column(train_set_sid, 'id')
test_set_sid = split_column(test_set_sid, 'id')

# merge senders with our dataframe
df = pd.merge(df, pd.concat([train_set_sid, test_set_sid]), how='outer', on='id')

# display process advancement
start = time()
print('Feature engineering before CV...')

# some mails are empty, we need to prevent nan
df['content'].fillna("", inplace=True)

# extract some features from datetime
df["datetime"] = pd.to_datetime(df["datetime"])
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour

# fill nan with median
for col in ['year', 'month', 'day', 'hour']:
    df[col].fillna(df[col].median(), inplace=True)

# create a feature that gave the sorted position of each mail by datetime
df.sort_values(['datetime']).reset_index(drop=True, inplace=True)
df['datetime_ix'] = range(1, len(df) + 1)

# extract length of the mail as a feature
df['mail_length'] = df['content'].map(len)

# load graph informations
girvan_newman_clusters = pickle.load(open('graph/girvan_newman_clusters', 'rb'))
clustering_nodes_coef = pickle.load(open('graph/clustering_nodes_coef', 'rb'))
kernighan_bisection = pickle.load(open('graph/kernighan_bisection', 'rb'))
nodes_triangles_nb = pickle.load(open('graph/nodes_triangles_nb', 'rb'))
louvain_clusters = pickle.load(open('graph/louvain_clusters', 'rb'))
kclique_clusters = pickle.load(open('graph/kclique_clusters', 'rb'))
square_clusters = pickle.load(open('graph/square_clusters', 'rb'))
nodes_pagerank = pickle.load(open('graph/nodes_pagerank', 'rb'))

# create graph clique features
for k, nodes in kclique_clusters.items():
    column = str(k) + '_clique'
    df[column] = df.sender.map(lambda x: 1 if x in nodes else 0)

# create kernighan bisection features
section1 = list(kernighan_bisection[0])
section2 = list(kernighan_bisection[1])
df['kernighan_sec1'] = df.sender.map(lambda x: 1 if x in section1 else 0)
df['kernighan_sec2'] = df.sender.map(lambda x: 1 if x in section2 else 0)

# create girvan newman communitiy features
for ix, community in enumerate(girvan_newman_clusters):
    column = 'girvan_newman_com' + str(ix)
    df[column] = df.sender.map(lambda x: 1 if x in community else 0)

# create clustering coef, louvain/square cluster triangles nb, pagerank features
df['clustering_coef'] = df.sender.map(clustering_nodes_coef)
df['louvain_cluster'] = df.sender.map(louvain_clusters)
df['square_cluster'] = df.sender.map(square_clusters)
df['triangles_nb'] = df.sender.map(nodes_triangles_nb)
df['pagerank'] = df.sender.map(nodes_pagerank)

# fill nan with median
for col in ['clustering_coef', 'louvain_cluster', 'square_cluster', 'triangles_nb', 'pagerank']:
    df[col].fillna(df[col].median(), inplace=True)

# get senders that send at least 75 mails
sender_count = df[df.recipients.notnull()].sender.apply(pd.Series, 1).stack()\
                                          .value_counts()

nb_sender = len(sender_count[sender_count >= 75])
big_senders = list(sender_count[:nb_sender].index)

# replace senders that send less than 75 mails by small_sender
df['recode_sender'] = df['sender'].map(lambda x: x if x in big_senders else 'small_sender')

# binarize senders
df = pd.get_dummies(data=df, columns=['recode_sender'])

# split df in train and test
train = df[df.recipients.notnull()].copy()
test = df[df.recipients.isnull()].copy()

# some mails have duplicated recipients, we need to clean that
train['recipients'] = train.recipients.str.split(' ').map(lambda x: ' '.join(list(set(x))))

# get recipients that received at least 11 mails
recip_count = train.recipients.str.split(' ').apply(pd.Series, 1).stack()\
                   .astype(str).value_counts()
                             
nb_recip = len(recip_count[recip_count >= 11])
big_recipients = list(recip_count[:nb_recip].index)

# define a function that delete recipients not in the list we have just created
def clean_recipients(recip_list):
    cleaned_recip = [recip for recip in recip_list if recip in big_recipients]
    if len(cleaned_recip) == 0:
        return np.nan
    else:
        return ' '.join(cleaned_recip)


# delete recipients that received less than 11 mails
train['recipients'] = train['recipients'].str.split(' ').apply(clean_recipients)

# remerge train and test
df = pd.concat([train, test])

# fill recipients nan
df['recipients'].fillna('no_recip', inplace=True)

# define a function that returns nth last mail's datetime index for the mail's sender
def nth_last_dt_ix(row, n):
    last_dt_ix = row.datetime_ix
    for _ in range(n):
        last_dt_ix = df[(df['datetime_ix'] < last_dt_ix) & (df['sender'] == row.sender)]\
                     .datetime_ix.max()
    if pd.isnull(last_dt_ix):
        last_dt_ix = 0
    return last_dt_ix


# define a function that returns nth last mail's recipients for the mail's sender
def nth_last_recips(row, n):
    last_dt_ix = nth_last_dt_ix(row, n)
    # if there is no previous mail from this sender, fix last_dt_ix to 7649.
    # like this, we will return no recip
    if last_dt_ix == 0:
        last_dt_ix = 7649
    last_recips = df.loc[df['datetime_ix'] == last_dt_ix, 'recipients'].tolist()[0]
    return last_recips


# init sklearn multi-label binarizer
mlb = MultiLabelBinarizer()

# create features for the last 3 mails of each mail's sender
for n, name in zip(range(1, 4), ['1st', '2nd', '3rd']):
    # create the nth previous last datetime index features for each mails
    col = name + '_last_dt_ix'
    df[col] = df.apply(lambda row: nth_last_dt_ix(row, n), axis=1)
    # add last mail's recipients as a feature in multilabel binarized format
    last_recips = df.apply(lambda row: nth_last_recips(row, n), axis=1)
    last_recips = pd.DataFrame(mlb.fit_transform(last_recips.str.split(' ')),
                               columns=[name + '_last_recip_' + recip for recip in mlb.classes_.tolist()])
    df = pd.concat([df, last_recips], axis=1)


# reverse no recip by nan
df['recipients'] = df['recipients'].map(lambda x: np.nan if x == 'no_recip' else x)

# resplit df in train and test
train = df[:len(train)].copy()
test = df[-len(test):].copy()

# delete rows with no recipients
train = train[train.recipients.notnull()].copy()

# encode recipients
le = LabelEncoder()
le.fit(big_recipients)
train['recipients'] = train['recipients'].str.split(' ').apply(le.transform)

# transform recipients to string with ids separated by one space
train['recipients'] = train['recipients'].map(lambda x: ' '.join(map(str, x)))

# get the labels list and prepare a sender dict where senders are the keys and
# the values are lists of their recipients labels (they can have multiple)
labels = le.classes_.tolist()
sender_dict = {}

for sender in list(sender_count.index):
    sender_dict[sender] = [recip_label
                           for recip_label, recip in enumerate(labels)
                           if sender.split('@')[0] in recip]

# be careful with short mails that could match with wrong labels
sender_dict['state@univ-tlse3.fr'] = []

# print time needed
print('took %s min %s sec\n' % (int((time() - start)/60),
                              round((time() - start) - int((time() - start)/60)*60, 1)))

# define a function that clean proba by writing to zero the proba that the
# sender is predicted
def zero_senders_proba(proba):
    for row, sender in zip(proba, test_sender):
        if len(sender_dict[sender]) > 0:
            for label in sender_dict[sender]:
                row[label] = 0
    return proba


#########                 STRATIFIED CROSS VALIDATION                 #########

# Sklearn stratified kfold doesn't support multilabel so we are going to
# monolabelize our recipients by spliting mails with multiple recipients into
# multiple rows.
# The score is a little bit optimistic because we don't take into account the
# recipients we have deleted when we apply our score function. 

# init a score list, stratified kfold, tfidf vectorizers, SVD, and LGBM clf objects
scores_lgbm = []
scores_rf = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

word_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=5,
                           norm='l2', sublinear_tf=True, max_features=5000,
                           token_pattern=r'\w{1,}')

char_vec = TfidfVectorizer(ngram_range=(2, 6), stop_words='english', min_df=5,
                           norm='l2', sublinear_tf=True, analyzer='char',
                           lowercase=False, token_pattern=r'\w{1,}')

svd = TruncatedSVD(n_components=100, algorithm='arpack', random_state=10)

lgbm_s = [
        LGBMClassifier(n_estimators=100, learning_rate=0.04, max_bin=128,
                       lambda_l2=0.25, random_state=10),
        LGBMClassifier(n_estimators=100, learning_rate=0.04, max_bin=128,
                       lambda_l2=0.25, random_state=4321),
        LGBMClassifier(n_estimators=100, learning_rate=0.04, max_bin=128,
                       lambda_l2=0.25, random_state=900),
        LGBMClassifier(n_estimators=100, learning_rate=0.04, max_bin=128,
                       lambda_l2=0.25, random_state=17),
        LGBMClassifier(n_estimators=100, learning_rate=0.04, max_bin=128,
                       lambda_l2=0.25, random_state=103),
        ]

rf_s = [
        RandomForestClassifier(n_estimators=500, random_state=2),
        RandomForestClassifier(n_estimators=500, random_state=0),
        RandomForestClassifier(n_estimators=500, random_state=30),
        RandomForestClassifier(n_estimators=500, random_state=28),
        RandomForestClassifier(n_estimators=500, random_state=136),
        ]

extra_trees_s = [
        ExtraTreesClassifier(n_estimators=200, random_state=2853),
        ExtraTreesClassifier(n_estimators=200, random_state=4),
        ExtraTreesClassifier(n_estimators=200, random_state=1),
        ExtraTreesClassifier(n_estimators=200, random_state=456),
        ExtraTreesClassifier(n_estimators=200, random_state=777),
        ]

clfs = [lgbm_s, rf_s, extra_trees_s]

# prepare a grid of blend weights (reduce the searching space for faster computation)
weights_lgbm = np.arange(0, 1.01, 0.02)
weights_rf = np.arange(0, 1.01, 0.02)
weights_grid = np.empty((1, 3), float)

# write weights combinations in the grid
for weight_lgbm in weights_lgbm:
    for weight_rf in weights_rf:
        if weight_lgbm + weight_rf <= 1:
            weights = np.array([[round(weight_lgbm, 2),
                                 round(weight_rf, 2),
                                 round(1 - weight_lgbm - weight_rf, 2)]])
            
            weights_grid = np.append(weights_grid, weights, axis=0)

# init a dictionnary that will store scores for each weights in the grid
scores_blend = {str(weights): [] for weights in [weights.tolist() for weights in weights_grid]}

# split recipients into multiple rows (we fit our model with mono-label)
train = split_column(train.copy(), 'recipients', replace=False)

# get tfidf for all mails
words = word_vec.fit_transform(train['content'])
chars = char_vec.fit_transform(train['content'])
chars = svd.fit_transform(chars)


# start cross validation
for train_ix, test_ix in skf.split(X=train, y=train['split_recipients']):
    
    # display cv advancement
    start = time()
    print('Performing fold %s...' % str(len(scores_lgbm) + 1))

    # get train and test folds
    xtrain = train.iloc[train_ix]
    xtest = train.iloc[test_ix].drop_duplicates(subset='id')
    ytrain = xtrain['split_recipients']
    ytest = xtest['recipients']
    
    # to compute mapk score we need to pass lists of lists in parameters
    ytest = [list(map(int, elem.split(' '))) for elem in ytest]
    
    # put test sender aside
    test_sender = xtest.sender.tolist()
    
    # join tfidf with numerical features
    to_drop = ['content', 'datetime', 'id', 'recipients', 'split_recipients',
               'sender']
    
    xtrain = np.concatenate((words[train_ix].toarray(), chars[train_ix],
                             xtrain.drop(to_drop, axis=1).values), axis=1)
    
    xtest = np.concatenate((words[xtest.index].toarray(), chars[xtest.index],
                            xtest.drop(to_drop, axis=1).values), axis=1)
    
    # fit, predict, and store proba predicted by each classifier
    probas = []
    
    for clf_s in clfs:
        proba_clf_s = 0
        for clf in clf_s:
            clf.fit(xtrain, ytrain)
            proba_clf_s += zero_senders_proba(clf.predict_proba(xtest))
        proba_clf_s = proba_clf_s/len(clf_s)
        probas.append(proba_clf_s)
    
    # compute and append scores
    for weights in weights_grid:
        proba_blend = 0
        for weight, proba in zip(weights, probas):
            proba_blend += weight*proba
            pred_blend = [sorted(range(len(row)), key=lambda i: row[i],
                                 reverse=True)[:10] for row in proba_blend]
            
        scores_blend[str(weights.tolist())].append(mapk(ytest, pred_blend, k=10))
        
    # compute and append lgbm and rf scores
    pred_lgbm = [sorted(range(len(row)), key=lambda i: row[i],
                        reverse=True)[:10] for row in probas[0]]
    
    pred_rf = [sorted(range(len(row)), key=lambda i: row[i],
                        reverse=True)[:10] for row in probas[1]]
    
    scores_lgbm.append(mapk(ytest, pred_lgbm, k=10))
    scores_rf.append(mapk(ytest, pred_rf, k=10))
    
    # print time needed to perform the fold
    print('took %s min %s sec' % (int((time() - start)/60),
                                  round((time() - start) - int((time() - start)/60)*60, 1)))


# compute CV score/std for each blend
for key, value in scores_blend.items():
    scores_blend[key] = [round(np.mean(value), 4), round(np.std(value), 4)]

# get the ten best blends
best_blends = sorted(scores_blend, key=scores_blend.get, reverse=True)[:10]

# print CV results
print('\nBest blends:')
for ix, weights in enumerate(best_blends):
    print('%s. CV score mapk = %s   std = %s   weights = %s' %
          (ix + 1, scores_blend[weights][0], scores_blend[weights][1], weights))

print('\nLgbm:')
print('CV score mapk = %s' % round(np.mean(scores_lgbm), 4))
print('          std = %s' % round(np.std(scores_lgbm), 4))

print('\nRf:')
print('CV score mapk = %s' % round(np.mean(scores_rf), 4))
print('          std = %s' % round(np.std(scores_rf), 4))
