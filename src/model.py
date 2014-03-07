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
    # Pull in perviously transformed data
    df = pd.read_csv('../data/transformed_features/submission_df.csv')
    df_train = pd.read_csv('../data/transformed_features/training_df.csv')

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
    # print '\nwriting CSV output...'
    # df[['id', 'pred']].to_csv('../submissions/output.csv', index=False)



if __name__ == '__main__':
    main()
