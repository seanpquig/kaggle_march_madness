"""
Script for preparing and transforming features from given data sets
"""

### IMPORT MODULES
import pandas as pd
import numpy as np
from sklearn import linear_model, feature_selection, metrics


# Get regular season data
df = pd.read_csv('../data/regular_season_results.csv')
df['wteam'] = df['wteam'].apply(str)
df['lteam'] = df['lteam'].apply(str)
wins = pd.DataFrame(df.groupby(['season', 'wteam']).size(), columns = ['wins'])
losses = pd.DataFrame(df.groupby(['season', 'lteam']).size(), columns = ['losses'])
win_loss_df = pd.merge(wins, losses, left_index=True, right_index=True, how='outer')
win_loss_df = win_loss_df.fillna(0)


# Get sample submissions
df = pd.read_csv('../data/sample_submission.csv')
del df['pred']

# Get lookup keys for submission
df['season'] = [i[0] for i in df.id]
df['team1'] = [i[2:5] for i in df.id]
df['team2'] = [i[6:] for i in df.id]

# Lookup win/loss data
df['team1_wins'] = [win_loss_df.loc[(df.season[i], df.team1[i])][0] for i in xrange(len(df))]
df['team1_losses'] = [win_loss_df.loc[(df.season[i], df.team1[i])][1] for i in xrange(len(df))]
df['team2_wins'] = [win_loss_df.loc[(df.season[i], df.team2[i])][0] for i in xrange(len(df))]
df['team2_losses'] = [win_loss_df.loc[(df.season[i], df.team2[i])][1] for i in xrange(len(df))]

# Setup result/output variable
tourney_results_df = pd.read_csv('../data/tourney_results.csv')
tourney_results = {}
for row in xrange(len(tourney_results_df)):
    season = tourney_results_df['season'][row]
    wteam = tourney_results_df['wteam'][row]
    lteam = tourney_results_df['lteam'][row]
    if wteam < lteam:
        tourney_results[(season, str(wteam), str(lteam))] = 1
    elif wteam > lteam:
        tourney_results[(season, str(lteam), str(wteam))] = 0
    else:
        print 'same team.'

results = []
for row in xrange(len(df)):
    try:
        results.append(tourney_results[(df.season[row], df.team1[row], df.team2[row])])
    except KeyError:
        results.append(None)

df['results'] = results
# Retrict to real games
df_test = df[ df['results'].notnull() ]


### SETUP FEATURES AND OUTPUT VARIABLE
features = df_test[['team1_wins']]
y_vals = df_test.results.values.astype(int)


### CREATE MODEL
model = linear_model.LogisticRegression().fit(features, y_vals)
y_pred = model.predict(features)
y_pred_probs = model.predict_proba(features)

print '\n', 'features: ', features.columns.values
print '\n', 'p-vals: ', feature_selection.univariate_selection.f_classif(features, y_vals)[1]
print '\n', 'coefs: ', model.coef_
print '\n', 'mean-acc: ', model.score(features, y_vals)

print '\n', 'conf. matrix: ', '\n', metrics.confusion_matrix(y_vals, y_pred)
print '\n', 'clasif. report: ', '\n', metrics.classification_report(y_vals, y_pred)
fpr, tpr, thresholds = metrics.roc_curve(y_vals, y_pred, pos_label=1)
print 'AUC: ', metrics.auc(fpr, tpr)

print 'LogLoss:  ', metrics.log_loss(y_vals, y_pred_probs)


### OUTPUT PREDICTIONS
features = df[['team1_wins']]
y_pred_probs = model.predict_proba(features)
df['pred'] = y_pred_probs.T[1]
df[['id', 'pred']].to_csv('../submissions/output.csv', index=False)


