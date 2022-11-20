#!/usr/bin/python3

import sqlite3
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

training_set_size = 0.8

def fix_outliers(dframe):
    ColumnNames=dframe.columns

    for j in ColumnNames:
        try:
            xy=dframe[j]
            mydata=pd.DataFrame()
            updated=[]
            Q1,Q3=np.percentile(xy,[25,75])
            IQR=Q3-Q1
            minimum=Q1-1.5*IQR
            maximum=Q3+1.5*IQR
            for i in xy:
                if(i>maximum):
                    i=maximum
                    updated.append(i)
                elif(i<minimum):
                    i=minimum
                    updated.append(i)
                else:
                    updated.append(i)
            dframe[j]=updated
        except:
            continue
    return dframe

def calculate_accuracy_lms(company, k_fold_val = False, train_set = None, test_set = None):
    if k_fold_val is False:
        if company is None:
            print ("\nNo company name supplied. Exiting.")
            quit()   
        
        print ('\nUsing the Least Mean Squares algorithm.')

        print ('\nCompany: ', company)   
        
        conn = sqlite3.connect('database.sqlite')
    
        query = 'SELECT CASE WHEN (home_team_goal - away_team_goal) > 0 THEN "H" WHEN (home_team_goal - away_team_goal) < 0 THEN "A" ELSE "D" END AS result, ' + company + 'H, ' + company + 'A, ' + company + 'D FROM Match WHERE NOT (B365H IS NULL OR B365A IS NULL OR B365D IS NULL OR BWH IS NULL OR BWA IS NULL OR BWD IS NULL OR IWH IS NULL OR IWA IS NULL OR IWD IS NULL OR LBH IS NULL OR LBA IS NULL OR LBD IS NULL)'
        df = pd.read_sql_query(query, conn)
        conn.close()
        df = fix_outliers(df)
        df_shuffled=df.sample(frac=1).reset_index(drop=True)  

        rowcount = math.floor(len(df) * training_set_size)
        train_set = df_shuffled.iloc[:rowcount, :]
        test_set = df_shuffled.iloc[(rowcount + 1 ):,:]

    X = np.empty((0,3))
    y = np.empty((0,3))
    w = np.array([[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]])
     
    for i, row in train_set.iterrows():
        X = np.append(X, [[row[1], row[2], row[3]]], axis=0)
        if row[0] == 'H': 
            y = np.append(y, [[1., -1., -1.]], axis=0)
        elif row[0] == 'D':
            y = np.append(y, [[-1., 1., -1.]], axis=0)
        else:
            y = np.append(y, [[-1., -1., 1.]], axis=0)
        error = y[i] - (w.T).dot(X[i])
        w_delta = 0.001 * np.dot(np.array([error]).T, np.array([X[i]]))

        w += w_delta

    hit = 0
    miss = 0

    t = ['H', 'D', 'A']
    for i, row in test_set.iterrows():
        r = np.array([row[1], row[2], row[3]]).dot(w)
        c = np.where( r == r.max())[0][0]

        if c == 0 and row[0] == 'H':
            hit += 1
        elif c == 1 and row[0] == 'D':
            hit += 1
        elif c == 2 and row[0] == 'A':
            hit += 1
        else:
            miss += 1

    if k_fold_val is False:
        print("hit: "+str(round((hit / len(test_set) * 100),3))+"%.")
        print("miss: "+str(round( (miss / len(test_set) * 100),3))+"%.")

    return (hit / len(test_set) * 100) 

def calculate_accuracy_ls(company, k_fold_val = False, train_set = None, test_set = None):

    if k_fold_val is False:
        if company is None:
            print ("\nNo company name supplied. Exiting.")
            quit()   
       
        print ('\nUsing the Least Squares algorithm.')

        print('\nCompany: ', company)   
        
        conn = sqlite3.connect('database.sqlite')
    
        query = 'SELECT CASE WHEN (home_team_goal - away_team_goal) > 0 THEN "H" WHEN (home_team_goal - away_team_goal) < 0 THEN "A" ELSE "D" END AS result, ' + company + 'H, ' + company + 'A, ' + company + 'D FROM Match WHERE NOT (B365H IS NULL OR B365A IS NULL OR B365D IS NULL OR BWH IS NULL OR BWA IS NULL OR BWD IS NULL OR IWH IS NULL OR IWA IS NULL OR IWD IS NULL OR LBH IS NULL OR LBA IS NULL OR LBD IS NULL)'
        df = pd.read_sql_query(query, conn)
        conn.close()
        df = fix_outliers(df)
        df_shuffled=df.sample(frac=1).reset_index(drop=True)  

        rowcount = math.floor(len(df) * training_set_size)
        train_set = df_shuffled.iloc[:rowcount, :]
        test_set = df_shuffled.iloc[(rowcount + 1 ):,:]

    X = np.array(np.zeros((len(train_set),4)))
    y = np.array(np.zeros((len(train_set),3)))

    for i, row in train_set.iterrows():
        X[i] = [row[1], row[2], row[3], 1]
        if row[0] == 'H':
            y[i] = [1, -1, -1]
        elif row[0] == 'D':
            y[i] = [-1, 1, -1]
        else:
            y[i] = [-1, -1, 1]

    Xtranspose = X.T
    dotProduct = Xtranspose.dot(X)
    inverse = np.linalg.pinv(dotProduct)
    A = inverse.dot(Xtranspose)
    w = A.dot(y)

    hit = 0
    miss = 0

    for i, row in test_set.iterrows():
        r = np.array([row[1], row[2], row[3], 1]).dot(w)
        c = np.where(r == r.max())[0][0]
        if c == 0 and row[0] == 'H':
            hit += 1
        elif c == 1 and row[0] == 'D':
            hit += 1
        elif c == 2 and row[0] == 'A':
            hit += 1
        else:
            miss += 1

    if k_fold_val is False:
        print("hit: "+str(round((hit / len(test_set) * 100),3))+"%.")
        print("miss: "+str(round( (miss / len(test_set) * 100),3))+"%.")
    
    return (hit / len(test_set) * 100) 

def mlp(k_fold_val = False, train_set = None, test_set = None):
    if k_fold_val is False:
        print('\nRetrieving data from database...\n')
        conn = sqlite3.connect('database.sqlite')

        query = 'SELECT CASE WHEN (home_team_goal - away_team_goal) > 0 THEN "H" WHEN (home_team_goal - away_team_goal) < 0 THEN "A" ELSE "D" END AS result,  Home_Team.buildUpPlaySpeed, Home_Team.buildUpPlayPassing, Home_Team.chanceCreationPassing, Home_Team.chanceCreationCrossing, Home_Team.chanceCreationShooting, Home_Team.defencePressure, Home_Team.defenceAggression, Home_Team.defenceTeamWidth, Away_Team.buildUpPlaySpeed, Away_Team.buildUpPlayPassing, Away_Team.chanceCreationPassing, Away_Team.chanceCreationCrossing, Away_Team.chanceCreationShooting, Away_Team.defencePressure, Away_Team.defenceAggression, Away_Team.defenceTeamWidth, B365H, B365A, B365D, BWH, BWA, BWD, IWH, IWA, IWD, LBH, LBA, LBD FROM Match INNER JOIN Team_Attributes Home_Team ON Home_Team.team_api_id = Match.home_team_api_id INNER JOIN Team_Attributes Away_Team ON Away_Team.team_api_id = Match.away_team_api_id  WHERE NOT (B365H IS NULL OR B365A IS NULL OR B365D IS NULL OR BWH IS NULL OR BWA IS NULL OR BWD IS NULL OR IWH IS NULL OR IWA IS NULL OR IWD IS NULL OR LBH IS NULL OR LBA IS NULL OR LBD IS NULL)'
        df = pd.read_sql_query(query, conn)
        conn.close()
        df = fix_outliers(df)
        df_shuffled=df.sample(frac=1).reset_index(drop=True)  

        rowcount = math.floor(len(df) * training_set_size)
        train_set = df_shuffled.iloc[:rowcount, :]
        test_set = df_shuffled.iloc[(rowcount + 1 ):,:]

    X = np.array(np.zeros((len(train_set),28)))
    y = np.array(np.zeros((len(train_set))))

    print('Preparing training data...\n')

    for i, row in train_set.iterrows():
        for j in range(1, len(row)):
            X[i][j - 1] = row[j]
        if row[0] == 'H':
            y[i] = 1
        elif row[0] == 'D':
            y[i] = 2
        else:
            y[i] = 3

    print('Training classifier...\n')

    clf = MLPClassifier(random_state=1)
    clf.out_activation_ = 'softmax'
    clf.fit(X, y)

    testX = np.array(np.zeros((len(test_set),28)))
    testy = np.array(np.zeros((len(test_set))))

    print('Preparing test data...\n')

    i = 0
    for w, row in test_set.iterrows():
        for j in range(1, len(row)):
            testX[i][j - 1] = row[j]
        if row[0] == 'H':
            testy[i] = 1
        elif row[0] == 'D':
            testy[i] = 2
        else:
            testy[i] = 3
        i += 1

    print('Testing...\n')

    hit = 0
    miss = 0

    i = 0
    for row in testX:
        result = clf.predict_proba([row])
        c = np.where(result[0] == result[0].max())[0][0]
        if c + 1 ==  testy[i]:
            hit += 1
        else:
            miss += 1
        i += 1

    if k_fold_val is False:
        print("hit: "+str(round((hit / len(test_set) * 100),3))+"%.")
        print("miss: "+str(round( (miss / len(test_set) * 100),3))+"%.")

    return (hit / len(test_set) * 100) 



def k_fold_validate(company = None, method = 'LMS'):
    if method == 'LMS':
        print ('\nUsing the Least Mean Squares algorithm.')
    elif method == 'LS':
        print ('\nUsing the Least Squares algorithm.')
    else:
        print ('\nUsing Multi-Layer Perceptron.')


    folds=10 
    conn = sqlite3.connect('database.sqlite')
    
    if(method == 'MLP'):
        query = 'SELECT CASE WHEN (home_team_goal - away_team_goal) > 0 THEN "H" WHEN (home_team_goal - away_team_goal) < 0 THEN "A" ELSE "D" END AS result,  Home_Team.buildUpPlaySpeed, Home_Team.buildUpPlayPassing, Home_Team.chanceCreationPassing, Home_Team.chanceCreationCrossing, Home_Team.chanceCreationShooting, Home_Team.defencePressure, Home_Team.defenceAggression, Home_Team.defenceTeamWidth, Away_Team.buildUpPlaySpeed, Away_Team.buildUpPlayPassing, Away_Team.chanceCreationPassing, Away_Team.chanceCreationCrossing, Away_Team.chanceCreationShooting, Away_Team.defencePressure, Away_Team.defenceAggression, Away_Team.defenceTeamWidth, B365H, B365A, B365D, BWH, BWA, BWD, IWH, IWA, IWD, LBH, LBA, LBD FROM Match INNER JOIN Team_Attributes Home_Team ON Home_Team.team_api_id = Match.home_team_api_id INNER JOIN Team_Attributes Away_Team ON Away_Team.team_api_id = Match.away_team_api_id  WHERE NOT (B365H IS NULL OR B365A IS NULL OR B365D IS NULL OR BWH IS NULL OR BWA IS NULL OR BWD IS NULL OR IWH IS NULL OR IWA IS NULL OR IWD IS NULL OR LBH IS NULL OR LBA IS NULL OR LBD IS NULL)'
    else:
        query = 'SELECT CASE WHEN (home_team_goal - away_team_goal) > 0 THEN "H" WHEN (home_team_goal - away_team_goal) < 0 THEN "A" ELSE "D" END AS result, ' + company + 'H, ' + company + 'A, ' + company + 'D FROM Match WHERE NOT (B365H IS NULL OR B365A IS NULL OR B365D IS NULL OR BWH IS NULL OR BWA IS NULL OR BWD IS NULL OR IWH IS NULL OR IWA IS NULL OR IWD IS NULL OR LBH IS NULL OR LBA IS NULL OR LBD IS NULL)'
    df = pd.read_sql_query(query, conn)
    conn.close()
    df = fix_outliers(df)
    df_shuffled=df.sample(frac=1).reset_index(drop=True)

    hit_results=[]

    sets = {}
    key = 0
    step = round(1/folds,2)
    total = round(1/folds,2)
    prev = 0
    while total <= 1:
        rowcount = math.floor(len(df) * total)
        sets[key] = df_shuffled.iloc[prev:rowcount, :] 
        total += step
        prev = rowcount + 1
        key += 1
   
    for dict_key in range(0,folds):
        if method == 'MLP':
            train_set = pd.DataFrame(columns=df.columns)
        else:
            train_set = pd.DataFrame(columns=['result',company+'H',company+'D',company+'A'])
        leftover_sets = dict(sets)
        leftover_sets.pop(dict_key)
        for frame in leftover_sets.values():
            train_set = train_set.append(frame, ignore_index=True)
        if method == 'MLP':
            result = mlp(True, train_set, sets[dict_key])
        elif method == 'LMS':
            result = calculate_accuracy_lms(company, True, train_set, sets[dict_key])
        else:
            result = calculate_accuracy_ls(company, True, train_set, sets[dict_key])
        hit_results.append(result)	
        print('Result for fold', str(dict_key + 1) + ':', result)

    if method == 'MLP':
        print ("\n"+str(folds)+"-fold cross-validated accuracy for Multi-Layer Perceptron:",str(round(sum(hit_results)/len(hit_results),2))+"%.\n")
    else:
        print ("\n"+str(folds)+"-fold cross-validated accuracy for "+company+" using "+str(method)+":",str(round(sum(hit_results)/len(hit_results),2))+"%.\n")
    return(round(sum(hit_results)/len(hit_results),2))

def most_accurate_betting_company(algorithm):
    print('\nFinding most accurate betting company using the', algorithm, 'algorithm...')
    companies = ['B365', 'BW', 'IW', 'LB']
    results = []
    for company in companies:
        print('\n--------------------------------\n')
        print('Company:', company,'\n')
        results.append(k_fold_validate(company,algorithm))
    print('Using the', algorithm, 'algorithm, the most accurate betting company is', companies[results.index(max(results))])

#calculate_accuracy_ls('B365')

k_fold_validate(None, 'MLP')
#mlp()

#most_accurate_betting_company('LMS')
#most_accurate_betting_company('LS')
#k_fold_validate(method='MLP')'''
