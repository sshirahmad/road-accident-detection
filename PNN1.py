import numpy as np
from numpy import savetxt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from neupy.algorithms import PNN
from sklearn.manifold import MDS, Isomap
from keras.models import Model, Sequential
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras.layers import Bidirectional, LSTM, Dropout, Dense, Activation, LeakyReLU, Input, Conv1D, MaxPooling1D, RepeatVector, Flatten
from keras.optimizers import Adam, Adadelta, SGD, RMSprop
from keras.preprocessing.sequence import TimeseriesGenerator
from imblearn.over_sampling import SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek


def result_model(model,x_train,y_train,x_test,y_test,len_series,num_epochs,batch_size,num_features):



    class_weight1 = {0: 1., 1: 100.}
    steps1 = len(generator) // batch_size
    trained_model = model.fit_generator(generator, steps_per_epoch=steps1, epochs=num_epochs, validation_data=generator_test, class_weight=class_weight1)

    score = model.evaluate_generator(generator_test)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # plot
    plt.figure(3)
    plt.plot(trained_model.history['binary_accuracy'], linewidth=3)
    plt.plot(trained_model.history['val_binary_accuracy'], linewidth=3)
    plt.plot(trained_model.history['loss'], linewidth=3)
    plt.plot(trained_model.history['val_loss'], linewidth=3)
    plt.ylabel('Loss & Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['acc', 'val_acc', 'loss', 'val_loss'], loc=10)
    plt.grid(color='b', linestyle='--', linewidth=0.5)
    plt.show()

    test_data = []
    true_labels = []
    for k in range(len(generator_test)):
          x, y = generator_test[k]
          test_data.append(x)
          true_labels.append(y)

    test_data = np.array(test_data)
    test_data = test_data.reshape(len(test_data),len_series,num_features)
    len_test_data = len(test_data)
    true_labels = np.array(true_labels)
    true_labels = true_labels.reshape(len(true_labels))

    predicted_y = model.predict(test_data)
    predicted_crash = np.around(predicted_y)

    return predicted_crash, true_labels, len_test_data


def confusion_matrix(predicted_labels,true_labels,len_test,flex):

    # Confusion Matrix
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for j in range(len(predicted_labels)):
        if (true_labels[j] == 1) and (sum(predicted_labels[j - flex:j + flex]) > 0):
            tp = tp + 1
        elif (true_labels[j] == 1) and (sum(predicted_labels[j - flex:j + flex]) == 0):
            fn = fn + 1
        elif (true_labels[j] == 0) and (predicted_labels[j] == 0):
            tn = tn + 1
        elif (true_labels[j] == 0) and (predicted_labels[j] == 1) and (sum(true_labels[j - flex:j + flex]) > 0):
            tn = tn + 1
        elif (true_labels[j] == 0) and (predicted_labels[j] == 1) and (sum(true_labels[j - flex:j + flex]) == 0):
            fp = fp + 1

    print("length of test data:", len_test)
    if (tp + fn + fp + tn) != len_test:
        print("invalid confusion matrix")

    return tp, fn, tn, fp


def resampling(x_train,y_train,resampling):

    print("resampling training data using SVMSMOTE:")
    x_train_res, y_train_res = resampling.fit_resample(x_train, y_train.ravel())

    return x_train_res,y_train_res


def visualization(x_train,y_train,x_train_res,y_train_res,resampling):
    dir1 = "space_before_"+str(resampling)+".jpg"
    dir2 = "space_after_" + str(resampling) + ".jpg"

    # apply PCA to visualize feature space
    plt.figure(1)
    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_train)
    crash = x_train_pca[y_train == 1]
    non_crash = x_train_pca[y_train == 0]
    plt.scatter(crash[:, 0], crash[:, 1])
    plt.scatter(non_crash[:, 0], non_crash[:, 1], c='r')
    plt.title('Feature space before resampling')
    plt.legend(['crash', 'non_crash'])
    plt.show()
    plt.savefig(dir1)

    plt.figure(2)
    pca = PCA(n_components=2)
    x_train_res_pca = pca.fit_transform(x_train_res)
    crash = x_train_res_pca[y_train_res == 1]
    non_crash = x_train_res_pca[y_train_res == 0]
    plt.scatter(crash[:, 0], crash[:, 1])
    plt.scatter(non_crash[:, 0], non_crash[:, 1], c='r')
    plt.title('Feature space after resampling')
    plt.legend(['crash', 'non_crash'])
    plt.show()
    plt.savefig(dir2)

    return


# load data
dataset = pd.read_csv('E:\MSc\Crash Detection\Data\Site1\Mix 5 min\site1_5min.csv', index_col=[0], header=[0,1])
dataset1 = pd.read_csv('E:\MSc\Crash Detection\Data\Site2\Mix 5 min\site2_5min.csv', index_col=[0], header=[0,1])

# cleaning the rows whose flow or occupancy equal -1
dataset=dataset[~(dataset[[('6368','Flow'),('6368','Occupancy'),('6369','Flow'),('6369','Occupancy'),('6370','Flow'),('6370','Occupancy'),
                           ('6371','Flow'),('6371','Occupancy'),('6372','Flow'),('6372','Occupancy'),('6373','Flow'),('6373','Occupancy'),
                           ('6374','Flow'),('6374','Occupancy'),('6375','Flow'),('6375','Occupancy'),('6426','Flow'),('6426','Occupancy'),
                           ('6427','Flow'),('6427','Occupancy'),('6428','Flow'),('6428','Occupancy'),('6429','Flow'),('6429','Occupancy'),
                           ('6430','Flow'),('6430','Occupancy'),('6431','Flow'),('6431','Occupancy'),('6432','Flow'),('6432','Occupancy'),
                           ('6433','Flow'),('6433','Occupancy')]].isin([-1])).any(1)]
dataset = dataset.reset_index(drop=True)

dataset1=dataset1[~(dataset1[[('6381','Flow'),('6381','Occupancy'),('6382','Flow'),('6382','Occupancy'),('6383','Flow'),('6383','Occupancy'),
                           ('6384','Flow'),('6384','Occupancy'),('6385','Flow'),('6385','Occupancy'),('6386','Flow'),('6386','Occupancy'),
                           ('6387','Flow'),('6387','Occupancy'),('6388','Flow'),('6388','Occupancy'),('6413','Flow'),('6413','Occupancy'),
                           ('6414','Flow'),('6414','Occupancy'),('6415','Flow'),('6415','Occupancy'),('6416','Flow'),('6416','Occupancy'),
                           ('6417','Flow'),('6417','Occupancy'),('6418','Flow'),('6418','Occupancy'),('6419','Flow'),('6419','Occupancy'),
                           ('6420','Flow'),('6420','Occupancy')]].isin([-1])).any(1)]
dataset1 = dataset1.reset_index(drop=True)

#cleaning the rows whose flow is greater than 3000
dataset=dataset[~(dataset[[('6368','Flow'),('6369','Flow'),('6370','Flow'),('6371','Flow'),('6372','Flow'),('6373','Flow'),('6374','Flow'),('6375','Flow'),
                           ('6426','Flow'),('6427','Flow'),('6428','Flow'),('6429','Flow'),('6430','Flow'),('6431','Flow'),('6432','Flow'),('6433','Flow')]].gt(3000)).any(1)]
dataset = dataset.reset_index(drop=True)

dataset1=dataset1[~(dataset1[[('6381','Flow'),('6382','Flow'),('6383','Flow'),('6384','Flow'),('6385','Flow'),('6386','Flow'),('6387','Flow'),('6388','Flow'),
                           ('6413','Flow'),('6414','Flow'),('6415','Flow'),('6416','Flow'),('6417','Flow'),('6418','Flow'),('6419','Flow'),('6420','Flow')]].gt(3000)).any(1)]
dataset1 = dataset1.reset_index(drop=True)

# separate labels and features and extract indexes of accidents
labels = dataset['labels']
labels = np.array(labels)
labels = labels.reshape(len(labels))
features = dataset.drop(columns='labels')
features = np.array(features)

labels1 = dataset1['labels']
labels1 = np.array(labels1)
labels1 = labels1.reshape(len(labels1))
features1 = dataset1.drop(columns='labels')
features1 = np.array(features1)

features_all = features # train two sites together
labels_all = labels

# normalize data
scaler = preprocessing.StandardScaler()

num_features = features_all.shape[1]
print("number of features after dimension reduction:",num_features)

# split the dataset using k-fold cross validation
kf = KFold(n_splits=5, random_state=None, shuffle=False)

x_dataset = []
y_dataset = []
x_dataset_test = []
y_dataset_test = []
for train_index, test_index in kf.split(features_all):
     print("TRAIN:", train_index, "TEST:", test_index)
     x_train, x_test = features_all[train_index], features_all[test_index]
     y_train, y_test = labels_all[train_index], labels_all[test_index]
     x_dataset.append(x_train)
     y_dataset.append(y_train)
     x_dataset_test.append(x_test)
     y_dataset_test.append(y_test)

x_dataset = np.array(x_dataset)
y_dataset = np.array(y_dataset)
x_dataset_test = np.array(x_dataset_test)
y_dataset_test = np.array(y_dataset_test)

#print('Number of non-crash data before oversampling in training data:', sum(y_train == 0))
#print('Number of crash data before oversampling in training data: {}\n'.format(sum(y_train == 1)))
#print('Number of non-crash data in test data:', sum(y_test == 0))
#print('Number of crash data in test data: {}\n'.format(sum(y_test == 1)))

sm1 = SMOTE()
sm2 = BorderlineSMOTE()
sm3 = SVMSMOTE()
sm4 = ADASYN()
sm5 = SMOTEENN()
sm6 = SMOTETomek()
dict_res = dict([('smote', sm1)])
names_res = np.array(['smote'])

flex = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
len_series = np.array([6])

sensitivity = np.zeros((len(x_dataset),len(names_res),len(len_series),len(flex)))
specificity = np.zeros((len(x_dataset),len(names_res),len(len_series),len(flex)))
accuracy = np.zeros((len(x_dataset),len(names_res),len(len_series),len(flex)))

for i in range(5):
    x_train = x_dataset[i]
    y_train = y_dataset[i]
    x_test = x_dataset_test[i]
    y_test = y_dataset_test[i]
    for j in range(1):

            # resampling training data
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            x_train_res, y_train_res = resampling(x_train, y_train, dict_res[names_res[j]])

            for k,length in enumerate(len_series):

                    # Create time series
                    generator = TimeseriesGenerator(x_train_res, y_train_res, length=length, reverse=True, batch_size=1)
                    generator_test = TimeseriesGenerator(x_test, y_test, length=length, reverse=True, batch_size=1)
                    train_data = []
                    train_labels = []
                    for n in range(len(generator)):
                        x, y = generator[n]
                        train_data.append(x)
                        train_labels.append(y)

                    train_data = np.array(train_data)
                    train_data = train_data.reshape(len(train_data), length*num_features)
                    train_labels = np.array(train_labels)
                    train_labels = train_labels.reshape(len(train_labels))

                    test_data = []
                    test_labels = []
                    for n in range(len(generator_test)):
                        x, y = generator_test[n]
                        test_data.append(x)
                        test_labels.append(y)

                    test_data = np.array(test_data)
                    test_data = test_data.reshape(len(test_data), length*num_features)
                    len_test = len(test_data)
                    test_labels = np.array(test_labels)
                    test_labels = test_labels.reshape(len(test_labels))

                    std = 1
                    pnn_network = PNN(std=std, batch_size=32, verbose=True)
                    pnn_network.train(train_data, train_labels)
                    predicted_labels = pnn_network.predict(test_data)
                    print("number of crashes: {}\n".format(sum(test_labels == 1)))

                    for l,f in enumerate(flex):
                            tp, fn, tn, fp = confusion_matrix(predicted_labels, test_labels, len_test, f)
                            accuracy[i, j, k, l] = (tp + tn) / (tp + tn + fp + fn)
                            sensitivity[i, j, k, l] = tp / (tp + fn)
                            specificity[i, j, k, l] = tn / (tn + fp)
                            print(
                                "accuracy of {}-th fold using resampling method {} with sequence length of {} with flexiblity {}:\n".format(
                                    i, names_res[j], length, f), accuracy[i, j, k, l])
                            print(
                                "sensitivity of {}-th fold using resampling method {} with sequence length of {} with flexiblity {}:\n".format(
                                    i, names_res[j], length, f), sensitivity[i, j, k, l])
                            print(
                                "specificity of {}-th fold using resampling method {} with sequence length of {} with flexiblity {}:\n".format(
                                    i, names_res[j], length, f), specificity[i, j, k, l])


acc_mean = accuracy.mean(axis=0)
sens_mean = sensitivity.mean(axis=0)
spec_mean = specificity.mean(axis=0)
print("accuracy of 5-folds:\n", acc_mean)
print("sensitivity of 5-folds:\n", sens_mean)
print("specificity of 5-folds:\n", spec_mean)







