"""
Script for preparing and transforming features from given data sets
"""

### IMPORT MODULES
import code
import pandas as pd
import numpy as np
from sklearn import linear_model, naive_bayes, svm, feature_selection, metrics, cross_validation
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier


def get_reg_season_data(df):
    """ Adds regular season data to input DataFrame. """
    
    reg_df = pd.read_csv('../data/regular_season_results.csv')
    reg_df['wteam'] = reg_df['wteam'].apply(str)
    reg_df['lteam'] = reg_df['lteam'].apply(str)

    # Get team wins and losses by season
    win_df = pd.DataFrame(reg_df.groupby(['season', 'wteam']).size(), columns = ['wins'])
    loss_df = pd.DataFrame(reg_df.groupby(['season', 'lteam']).size(), columns = ['losses'])

    # Get team points for and points against by season
    win_pts_df = reg_df.groupby(['season', 'wteam']).sum()[['wscore', 'lscore']]
    win_pts_df.columns = ['PF_in_wins', 'PA_in_wins']
    loss_pts_df = reg_df.groupby(['season', 'lteam']).sum()[['lscore', 'wscore']]
    loss_pts_df.columns = ['PF_in_losses', 'PA_in_losses']

    # Pull data together
    wl_df = pd.merge(win_df, loss_df, left_index=True, right_index=True, how='outer')
    wl_df = pd.merge(wl_df, win_pts_df, left_index=True, right_index=True, how='outer')
    wl_df = pd.merge(wl_df, loss_pts_df, left_index=True, right_index=True, how='outer')
    wl_df = wl_df.fillna(0)

    # Pull win/loss data into core DataFrame
    df['team1_wins'] = [wl_df.loc[(df.season[i], df.team1[i])][0] for i in xrange(len(df))]
    df['team1_losses'] = [wl_df.loc[(df.season[i], df.team1[i])][1] for i in xrange(len(df))]
    df['team1_PF_in_wins'] = [wl_df.loc[(df.season[i], df.team1[i])][2] for i in xrange(len(df))] 
    df['team1_PA_in_wins'] = [wl_df.loc[(df.season[i], df.team1[i])][3] for i in xrange(len(df))]
    df['team1_PF_in_losses'] = [wl_df.loc[(df.season[i], df.team1[i])][4] for i in xrange(len(df))]
    df['team1_PA_in_losses'] = [wl_df.loc[(df.season[i], df.team1[i])][5] for i in xrange(len(df))]

    df['team2_wins'] = [wl_df.loc[(df.season[i], df.team2[i])][0] for i in xrange(len(df))]
    df['team2_losses'] = [wl_df.loc[(df.season[i], df.team2[i])][1] for i in xrange(len(df))]
    df['team2_PF_in_wins'] = [wl_df.loc[(df.season[i], df.team2[i])][2] for i in xrange(len(df))] 
    df['team2_PA_in_wins'] = [wl_df.loc[(df.season[i], df.team2[i])][3] for i in xrange(len(df))]
    df['team2_PF_in_losses'] = [wl_df.loc[(df.season[i], df.team2[i])][4] for i in xrange(len(df))]
    df['team2_PA_in_losses'] = [wl_df.loc[(df.season[i], df.team2[i])][5] for i in xrange(len(df))]

    # Calculated metrics
    df['team1_GP'] = df['team1_wins'] + df['team1_losses']
    df['team1_PF'] = df['team1_PF_in_wins'] + df['team1_PF_in_losses']
    df['team1_PA'] = df['team1_PA_in_wins'] + df['team1_PA_in_losses']
    df['team1_pt_diff'] = (df['team1_PF'] - df['team1_PA']) / df['team1_GP']

    df['team2_GP'] = df['team2_wins'] + df['team2_losses']
    df['team2_PF'] = df['team2_PF_in_wins'] + df['team2_PF_in_losses']
    df['team2_PA'] = df['team2_PA_in_wins'] + df['team2_PA_in_losses']
    df['team2_pt_diff'] = (df['team2_PF'] - df['team2_PA']) / df['team2_GP']

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
    """ Adds tourney results to input DataFrame as output variable. """

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


def print_model_stats(model, X, y, msg):
    print '\n', msg
    y_pred = model.predict(X)
    y_pred_probs = model.predict_proba(X)

    print '  conf. matrix: ', '\n ', metrics.confusion_matrix(y, y_pred)
    # print '\n', 'clasif. report: ', '\n', metrics.classification_report(y_test, y_pred)
    print '  AUC:  ', metrics.roc_auc_score(y, y_pred)
    print '  mean-acc: ', model.score(X, y)
    print '  LogLoss:  ', metrics.log_loss(y, y_pred_probs)


def print_cross_val_stats(model, features, labels):
    # Cross Validation
    AUCs, MAs, LogLosses = [], [], []
    np.random.seed(1234)
    for i in xrange(20):
        X_train, X_test, y_train, y_test = \
            cross_validation.train_test_split(features, labels, test_size=.35)

        y_pred_test = model.fit(X_train, y_train).predict(X_test)
        y_pred_probs_test = model.predict_proba(X_test)
        fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_pred_test, pos_label=1)

        AUCs.append(metrics.auc(fpr_test, tpr_test))
        MAs.append(model.score(X_test, y_test))
        LogLosses.append(metrics.log_loss(y_test, y_pred_probs_test))

    print '\nCross-validation:'
    print '  average AUC: ', np.mean(AUCs)
    print '  range AUC: ', min(AUCs), '-', max(AUCs)
    print '  average mean-acc:  ', np.mean(MAs)
    print '  range mean-acc:  ', min(MAs), '-', max(MAs)
    print '  average LogLoss:  ', np.mean(LogLosses)
    print '  range LogLoss:  ', min(LogLosses), '-', max(LogLosses)


def main():
    # set up core DataFrame objects for data transformation and analysis
    df = pd.read_csv('../data/sample_submission.csv')
    df_train = pd.read_csv('../data/training_labels.csv')
    del df['pred']

    # create lookup key columns
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

    # Restrict to real matchups and exculde play-in games
    df_train = df_train[ df_train['daynum'] > 135 ]
    df_test = df[ (df['result'].notnull()) & (df['daynum'] > 135) ]


    ### SETUP FEATURES AND DEPENDENT VARIABLE DATA
    features = ['team1_wins', 'team2_wins', 'team1_losses', 'team2_losses', 
                'team1_seed', 'team2_seed', 'team1_pt_diff', 'team2_pt_diff']
    X_train, y_train = df_train[features], df_train.result.values.astype(int)
    X_test, y_test = df_test[features], df_test.result.values.astype(int)

    
    ### CREATE MODEL AND RUN STATISTICS
    model = linear_model.LogisticRegression().fit(X_train, y_train)
    # model = RandomForestClassifier(n_estimators=50, max_depth = 3).fit(train_features, train_labels)
    # model = ExtraTreesClassifier(n_estimators=500, max_depth = 3).fit(X_train, y_train)
    # model = AdaBoostClassifier().fit(X_train, y_train)
    # model = svm.SVC(probability=True).fit(X_train, y_train)
    # model = naive_bayes.GaussianNB().fit(X_train, y_train)

    print '\n', 'features: ', features
    print 'p-vals: ', feature_selection.univariate_selection.f_classif(X_train, y_train)[1]
    print 'coefs: ', model.coef_

    print_cross_val_stats(model, X_train, y_train)
    print_model_stats(model, X_train, y_train, 'training set:')
    print_model_stats(model, X_test, y_test, 'scored set:')


    ### OUTPUT PREDICTIONS
    X_vals = df[features]
    y_pred_probs = model.predict_proba(X_vals)
    df['pred'] = y_pred_probs.T[1]
    # print '\nwriting csv output...'
    # df[['id', 'pred']].to_csv('../submissions/output.csv', index=False)


if __name__ == '__main__':
    main()
