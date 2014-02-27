"""
Script for preparing and transforming features from given data sets
"""

### IMPORT MODULES
import code
import pandas as pd
import numpy as np
from sklearn import linear_model, feature_selection, metrics


def get_reg_season_data(df):
    """ Adds regular season data to input DataFrame. """

    reg_df = pd.read_csv('../data/regular_season_results.csv')
    reg_df['wteam'] = reg_df['wteam'].apply(str)
    reg_df['lteam'] = reg_df['lteam'].apply(str)

    win_df = pd.DataFrame(reg_df.groupby(['season', 'wteam']).size(), columns = ['wins'])
    loss_df = pd.DataFrame(reg_df.groupby(['season', 'lteam']).size(), columns = ['losses'])
    wl_df = pd.merge(win_df, loss_df, left_index=True, right_index=True, how='outer')
    wl_df = wl_df.fillna(0)

    # pull win/loss data into core DataFrame
    df['team1_wins'] = [wl_df.loc[(df.season[i], df.team1[i])][0] for i in xrange(len(df))]
    df['team1_losses'] = [wl_df.loc[(df.season[i], df.team1[i])][1] for i in xrange(len(df))]
    df['team2_wins'] = [wl_df.loc[(df.season[i], df.team2[i])][0] for i in xrange(len(df))]
    df['team2_losses'] = [wl_df.loc[(df.season[i], df.team2[i])][1] for i in xrange(len(df))]

    return df


def get_tourney_results(df):
    """ Pulls tourney results into input df as output variable. """

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
            print 'There\'s a problem:  same team listed twice in a matchup.'

    results = []
    for row in xrange(len(df)):
        try:
            results.append(tourney_results[(df.season[row], df.team1[row], df.team2[row])])
        except KeyError:
            results.append(None)

    df['results'] = results
    return df


def main():
    # set up core DataFrame object for data transformation and analysis
    df = pd.read_csv('../data/sample_submission.csv')
    del df['pred']

    # break out lookup keys
    df['season'] = [i[0] for i in df.id]
    df['team1'] = [i[2:5] for i in df.id]
    df['team2'] = [i[6:] for i in df.id]

    # pull in regular season data
    df = get_reg_season_data(df)

    # pull in tourney results as output variable
    df = get_tourney_results(df)


    # Retrict to real games
    df_test = df[ df['results'].notnull() ]


    ### SETUP FEATURES AND OUTPUT VARIABLE
    features = ['team1_wins', 'team2_wins']
    x_vals = df_test[features]
    y_vals = df_test.results.values.astype(int)


    ### CREATE MODEL
    model = linear_model.LogisticRegression().fit(x_vals, y_vals)
    y_pred = model.predict(x_vals)
    y_pred_probs = model.predict_proba(x_vals)

    print '\n', 'features: ', x_vals.columns.values
    print 'p-vals: ', feature_selection.univariate_selection.f_classif(x_vals, y_vals)[1]
    print 'coefs: ', model.coef_

    print '\n', 'conf. matrix: ', '\n', metrics.confusion_matrix(y_vals, y_pred)
    # print '\n', 'clasif. report: ', '\n', metrics.classification_report(y_vals, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_vals, y_pred, pos_label=1)
    print '\n', 'AUC: ', metrics.auc(fpr, tpr)
    print 'mean-acc: ', model.score(x_vals, y_vals)
    print 'LogLoss:  ', metrics.log_loss(y_vals, y_pred_probs)


    ### OUTPUT PREDICTIONS
    # x_vals = df[features]
    # y_pred_probs = model.predict_proba(x_vals)
    # df['pred'] = y_pred_probs.T[1]
    # df[['id', 'pred']].to_csv('../submissions/output.csv', index=False)


if __name__ == '__main__':
    main()
