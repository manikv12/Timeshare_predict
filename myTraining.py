import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle

def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data)) #create a random array with our data inside
    test_set_size = int(len(data)*ratio) #Get the %age off data you want to slice for test
    test_indices = shuffled[:test_set_size] #Set last part of data for testing in array
    train_indices = shuffled[test_set_size:] #set first part of data for training in array
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    train, test = data_split(df,0.25)
    x_train = train[['time_value','resort_grp','num_deeds','household_income','ts_type']].to_numpy()
    x_test = test[['time_value','resort_grp','num_deeds','household_income','ts_type']].to_numpy()
    y_train = train[['target']].to_numpy().reshape(2382,)
    y_train = train[['target']].to_numpy().reshape(2382,)
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(x_train,y_train)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)

    # close the file
    file.close()




    #Code for inference
    inputFeatures = [10000,1,8,60000,1]
    infProb =clf.predict([[15000,3,5,70000,1]])
    print(infProb)
    if infProb[0] == 0:
        print('Not Interested!')
    else:
        print('Intrested!')