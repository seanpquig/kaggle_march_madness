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


def get_seeds(df):
    """ Adds tourney seed data to input DataFrame. """

    seeds_df = pd.read_csv('../data/tourney_seeds.csv')
    seeds_df['team'] = seeds_df['team'].apply(str)
    tourney_seeds = {}
    for i in xrange(len(seeds_df)):
        season = seeds_df['season'][i]
        region = seeds_df['seed'][i][0]
        seed = int(seeds_df['seed'][i][1:3])
        team = seeds_df['team'][i]
        tourney_seeds[(season, team)] = (region, seed)
    
    regions_t1, regions_t2 = [], []
    seeds_t1, seeds_t2 = [], []
    for i in xrange(len(df)):
        # lookup seeds for team1
        try:
            regions_t1.append(tourney_seeds[(df.season[i], df.team1[i])][0])
            seeds_t1.append(tourney_seeds[(df.season[i], df.team1[i])][1])
        except KeyError:
            regions_t1.append(None)
            seeds_t1.append(None)

        # lookup seeds for team2
        try:
            regions_t2.append(tourney_seeds[(df.season[i], df.team2[i])][0])
            seeds_t2.append(tourney_seeds[(df.season[i], df.team2[i])][1])
        except KeyError:
            regions_t2.append(None)
            seeds_t2.append(None)

    df['team1_region'], df['team2_region'] = regions_t1, regions_t2
    df['team1_seed'], df['team2_seed'] = seeds_t1, seeds_t2
    return df


def get_tourney_results(df):
    """ Pulls tourney results into input df as output variable. """

    tourney_results_df = pd.read_csv('../data/tourney_results.csv')
    tourney_results = {}
    for i in xrange(len(tourney_results_df)):
        season = tourney_results_df['season'][i]
        wteam = tourney_results_df['wteam'][i]
        lteam = tourney_results_df['lteam'][i]
        daynum = tourney_results_df['daynum'][i]
        if wteam < lteam:
            tourney_results[(season, str(wteam), str(lteam))] = (1, daynum)
        elif wteam > lteam:
            tourney_results[(season, str(lteam), str(wteam))] = (0, daynum)
        else:
            print 'There\'s a problem:  same team listed twice in a matchup.'

    results = []
    daynum = []
    for i in xrange(len(df)):
        try:
            results.append(tourney_results[(df.season[i], df.team1[i], df.team2[i])][0])
            daynum.append(tourney_results[(df.season[i], df.team1[i], df.team2[i])][1])
        except KeyError:
            results.append(None)
            daynum.append(None)

    df['result'] = results
    df['daynum'] = daynum
    return df



def main():
    # set up core DataFrame object for data transformation and analysis
    df = pd.read_csv('../data/sample_submission.csv')
    del df['pred']

    df_train = pd.read_csv('../data/training_labels.csv')
    df_train.columns = ['id', 'result_excel']

    # break out lookup keys
    df['season'], df_train['season'] = [i[0] for i in df.id], [i[0] for i in df_train.id]
    df['team1'], df_train['team1'] = [i[2:5] for i in df.id], [i[2:5] for i in df_train.id]
    df['team2'], df_train['team2'] = [i[6:] for i in df.id], [i[6:] for i in df_train.id]

    # pull in regular season data
    df = get_reg_season_data(df)
    df_train = get_reg_season_data(df_train)
    
    # get tourney seed data
    df = get_seeds(df)
    df_train = get_seeds(df_train)

    # pull in tourney results as output variable
    df = get_tourney_results(df)
    df_train = get_tourney_results(df_train)

    # calculated features
    # df['seed_diff'] = df['team1_seed'] - df['team2_seed']

    # Restrict to real games and exculde play-in games
    df_test = df[ df['result'].notnull() & (df['daynum'] > 135) ]


    ### SETUP FEATURES AND DEPENDENT VARIABLE
    features = ['team1_wins', 'team2_wins', 'team1_losses', 'team2_losses', 
                'team1_seed', 'team2_seed']
    X_test = df_test[features]
    y_test = df_test.result.values.astype(int)

    X_train = df_train[features]
    y_train = df_train.result.values.astype(int)


    ### CREATE MODEL
    model = linear_model.LogisticRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)

    print '\n', 'features: ', X_test.columns.values
    print 'p-vals: ', feature_selection.univariate_selection.f_classif(X_test, y_test)[1]
    print 'coefs: ', model.coef_

    print '\n', 'conf. matrix: ', '\n', metrics.confusion_matrix(y_test, y_pred)
    # print '\n', 'clasif. report: ', '\n', metrics.classification_report(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    print '\n', 'AUC: ', metrics.auc(fpr, tpr)
    print 'mean-acc: ', model.score(X_test, y_test)
    print 'LogLoss:  ', metrics.log_loss(y_test, y_pred_probs)


    ### OUTPUT PREDICTIONS
    x_vals = df[features]
    y_pred_probs = model.predict_proba(x_vals)
    df['pred'] = y_pred_probs.T[1]
    print '\nwriting csv output...'
    df[['id', 'pred']].to_csv('../submissions/output.csv', index=False)


if __name__ == '__main__':
    main()
