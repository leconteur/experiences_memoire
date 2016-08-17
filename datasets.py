import argparse
import os
from functools import partial

import ddp
import pandas
import json

from sklearn import utils

parser = argparse.ArgumentParser('Datasets creator',
                                 description='Creates the training and validation datasets for my memoire.')
parser.add_argument('filename', help='The filepath of the file containing the data from every experiment.')
parser.add_argument('testfrac', type=float, help='The fraction of the data that will be in the testset set.')
parser.add_argument('ddpFilterType', choices=['median', 'mean', 'ewma', 'workload'],
                    help='The type of filter used in the computation of the ddp')
parser.add_argument('ddpFilterWidth', type=int, help='The length of the filter used for the computation of the ddp.')
parser.add_argument('ddpThresholdType', choices=['yen', 'li'])
parser.add_argument('--output', default='datasets')


def split_group(group, split_frac):
    index = group.dropna().index
    group = pandas.Series(index=index, data=0.)
    try:
        split_pivot = index[int(len(index) * split_frac)]
        group[:split_pivot] = 1.
    except IndexError:  # if the group is empty, just return an empty serie
        pass
    return group


def save_dataset(data, params):
    train = data[data.testset == 0].drop(['testset'], axis=1)
    test = data[data.testset == 1].drop(['testset'], axis=1)

    filtered_reactiontime_name = "_".join(['reaction time', args.ddpFilterType, str(args.ddpFilterWidth)])
    target_columns = [filtered_reactiontime_name, u'ddp', u'reaction time']
    info_columns = [u'workload', u'taskname', u'timeofday', u'participant']
    bio_columns = list(set(train.columns) - set(target_columns) - set(info_columns))

    report = {'targets': target_columns, 'category_columns': info_columns, 'bio': bio_columns,
              'ddp_filter_type': args.ddpThresholdType, 'ddp_filter_width': args.ddpFilterWidth,
              'ddp_threshold_type': args.ddpThresholdType, 'TrainFrac': 1 - args.testfrac}


    os.makedirs(args.output)

    with open(os.path.join(args.output, 'info.json'), 'w') as f:
        json.dump(report, f)

    train_filepath = os.path.join(args.output, 'trainset.csv')
    test_filepath = os.path.join(args.output, 'testset.csv')
    train.to_csv(train_filepath, encoding='utf-8')
    test.to_csv(test_filepath, encoding='utf-8')
    return
def load_training_data(datasetdir, trainfile):
    with open(os.path.join(datasetdir, 'info.json')) as f:
        info = json.load(f)
    trainfile = os.path.join(datasetdir, trainfile)
    totaltrainset = pandas.read_csv(trainfile, index_col='Time', parse_dates=True).dropna()
    return totaltrainset, info


def process_data(trainset, info, context, shuffle):
    cols = info['bio'][:]
    if context == True:
        cols += ['timeofday', 'workload']
        trainset['timeofday'], _ = pandas.factorize(trainset.timeofday, sort=True)
        trainset['workload'], _ = pandas.factorize(trainset.workload, sort=True)
    training_data = trainset[cols].values.astype(float)
    training_target = trainset[['ddp']].values.ravel()
    training_label = trainset.index.to_period('5min')
    if shuffle == True:
        return utils.shuffle(training_data, training_target, training_label)
    else:
        return training_data, training_target, training_label


def create_sequence(X, y, labels, numsteps, batchsize):
    offset = X.shape[0] % (numsteps * batchsize)
    X = X[:-offset].reshape(-1, numsteps, X.shape[-1])
    y = y[:-offset].reshape(-1, numsteps)
    labels = labels[:-offset].values.reshape(-1, numsteps)
    return X, y, labels

if __name__ == "__main__":
    args = parser.parse_args()
    data = pandas.read_csv(args.filename, parse_dates=True, index_col='Time')
    data_with_ddp = ddp.ddp(data, 'reaction time', ['participant', 'taskname'], args.ddpFilterType, args.ddpFilterWidth,
                            args.ddpThresholdType)

    categories = ['participant', 'taskname', 'timeofday', 'workload']
    splitter = partial(split_group, split_frac=args.testfrac)
    data_with_ddp['testset'] = data_with_ddp.groupby(categories)['reaction time'].transform(splitter)

    #print(data_with_ddp.dropna())
    save_dataset(data_with_ddp, params=args)


