import os
import pickle

import numpy
import sklearn
from sklearn import grid_search
from sklearn import metrics
from sklearn import preprocessing, neighbors, svm, ensemble
from sklearn.cross_validation import LabelShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from tensorflow.contrib import skflow

from datasets import load_training_data, process_data


def get_classifier(classifier_type, input_shape):
    if classifier_type == 'knn':
        cls = make_pipeline(preprocessing.RobustScaler(), neighbors.KNeighborsClassifier())
        params = {'kneighborsclassifier__n_neighbors': [i for i in range(1, 100)]}
    elif classifier_type == 'svm':
        cls = make_pipeline(preprocessing.RobustScaler(), svm.SVC(class_weight='balanced'))
        params = {'svc__C': numpy.logspace(-3, 2, 600),
                  'svc__gamma': numpy.logspace(-3, 2, 600)}
    elif classifier_type == 'randomforest':
        cls = make_pipeline(preprocessing.RobustScaler(), ensemble.RandomForestClassifier(class_weight='balanced'))
        params = {'randomforestclassifier__n_estimators': [i for i in range(1, 100)]}
    elif classifier_type == 'rnn':
        print(input_shape)
        exit()
        cls = skflow.TensorFlowRNNClassifier(rnn_size=input_shape[0],
                                                n_classes=2, cell_type='lstm',
                                                num_layers=2, bidirectional=False, sequence_length=None,
                                                steps=60, optimizer='Adam', learning_rate=0.01,
                                                continue_training=True)
    elif classifier_type == "freq":
        cls = DummyClassifier(strategy='most_frequent')
        params = {}
    return cls, params


def save_model(classifier_type, participant, taskname, model_dir, classifier):
    model_filename = '{}-{}-{}.pkl'.format(classifier_type, participant, taskname)
    model_path = os.path.join(model_dir, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump(search, f)


def train(classifier_type, label_kfold, training_data, training_target):
    cls, params = get_classifier(classifier_type, training_data.shape)
    mcc = metrics.make_scorer(metrics.matthews_corrcoef)
    n_iter = 50 if classifier_type not in ['freq'] else 1
    search = grid_search.RandomizedSearchCV(cls, params, scoring=mcc, n_iter=n_iter, n_jobs=NJOBS, iid=False,
                                            cv=label_kfold, verbose=True, random_state=SEED)
    search.fit(training_data, training_target)
    print(search.best_params_, search.best_score_)
    return search


if __name__ == "__main__":
    NJOBS = 1
    context = False
    classifier_type = 'freq'
    datasetdir = 'datasets_begin'
    trainfile = 'trainset.csv'
    granularity = 'generalize_taskname'
    model_dir = os.path.join('basic_models', granularity)
    SEED = 1000
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    totaltrainset, info = load_training_data(datasetdir, trainfile)

    participant = 'all'
    taskname = 'all'
    isSequence = classifier_type == 'rnn'
    if granularity == 'all':
        training_data, training_target, training_label = process_data(totaltrainset, info, context, True)
        label_kfold = LabelShuffleSplit(training_label, n_iter=4, random_state=SEED)

        search = train(classifier_type, label_kfold, training_data, training_target)
        save_model(classifier_type, participant, taskname, model_dir, search)
    elif granularity == 'generalize_participant':
        for participant in totaltrainset.participant.unique():
            trainset = totaltrainset[totaltrainset.participant != participant]
            training_data, training_target, training_label = process_data(trainset, info, context, True)
            label_kfold = LabelShuffleSplit(training_label, n_iter=4, random_state=SEED)

            search = train(classifier_type, label_kfold, training_data, training_target)
            save_model(classifier_type, participant, taskname, model_dir, search)
    elif granularity == 'generalize_taskname':
        for taskname in totaltrainset.taskname.unique():
            trainset = totaltrainset[totaltrainset.taskname != taskname]
            training_data, training_target, training_label = process_data(trainset, info, context, True)
            label_kfold = LabelShuffleSplit(training_label, n_iter=4, random_state=SEED)

            search = train(classifier_type, label_kfold, training_data, training_target)
            save_model(classifier_type, participant, taskname, model_dir, search)
    else:
        for variable, trainset in totaltrainset.groupby([granularity]):
            if granularity == 'participant':
                participant = variable
            elif granularity == 'taskname':
                taskname = variable
            training_data, training_target, training_label = process_data(trainset, info, context, True)
            label_kfold = LabelShuffleSplit(training_label, n_iter=4, random_state=SEED)

            search = train(classifier_type, label_kfold, training_data, training_target)
            save_model(classifier_type, participant, taskname, model_dir, search)
