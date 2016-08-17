#! encoding=utf-8
from __future__ import print_function, unicode_literals

import glob
import itertools

import matplotlib

matplotlib.use('Agg')

import datasets
import numpy
import pandas
import os
import json

import pickle

from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

TRADUCTION = {
    'nback': 'N-Back', 'mental rotation': 'Rotation mentale', 'visual search': 'Recherche visuelle',
    'randomforest': unicode('Foret aleatoire'), 'taskname': 'Tâche', 'participant': 'Participant',
    'all': 'Tout', 'generalize_participant': 'Généralisation\npar participant',
    'generalize_taskname': 'Généralisation\npar tâche'
}

TRADUCTION_COLUMN = {'taskname': 'Tâche', 'granularity': 'Modélisation', 'classifier': 'Classifieur'}

granularity_str = TRADUCTION_COLUMN['granularity']


def evaluate_classfier(testset, info, cls_dict, context, groupby=None):
    predict = testset.groupby(['participant', 'taskname']).apply(lambda x: classify_group(x, cls_dict, info, context))
    testset['ddp_predict'] = predict.reset_index([0, 1], drop=True)
    # return testset.groupby(['participant', 'taskname']).apply(compute_mcc)
    if groupby is None:
        return compute_mcc(testset)
    else:
        return testset.groupby(groupby).apply(compute_mcc)


def classify_group(group, cls_dict, info, context):
    participant = group.iloc[0]['participant']
    task = group.iloc[0]['taskname']
    try:
        cls = cls_dict[(participant, task)]
        cls = cls.best_estimator_
        testing_data, _, _ = datasets.process_data(group, info, context, False)
        y_label = cls.predict(testing_data)
    except KeyError:
        y_label = numpy.nan
    return pandas.Series(data=y_label, index=group.index, name='ddp_predict')


def compute_mcc(data):
    data = data.dropna()
    ddp = data['ddp'].values
    ddp_predict = data['ddp_predict'].values
    return metrics.matthews_corrcoef(ddp, ddp_predict)


class Models(object):
    def __init__(self, model_dir, model_type):
        self.cls_dict = {}
        filename = '{}-*-*.pkl'.format(model_type)
        files = glob.glob(os.path.join(model_dir, filename))
        self.all_participant = False
        self.all_taskname = False
        for file in files:
            classifier_type, participant, taskname = file[:-4].split('-')
            if participant == 'all':
                self.all_participant = True
            if taskname == 'all':
                self.all_taskname = True
            with open(file, 'rb') as f:
                cls = pickle.load(f)
            self.cls_dict[(participant, taskname)] = cls

    def __getitem__(self, key):
        participant, taskname = key
        if self.all_participant:
            participant = 'all'
        if self.all_taskname:
            taskname = 'all'
        return self.cls_dict[(participant, taskname)]


def violin_basic(testset, info):
    granularities = ['taskname', 'participant', 'all', 'generalize_participant', 'generalize_taskname']
    model_types = ['randomforest', 'svm', 'knn']  # + ['freq']
    if os.path.exists('evals_violin.pkl'):
        with open('evals_violin.pkl', 'rb') as f:
            evals = pickle.load(f)
    else:
        evals = eval_violin(granularities, info, model_types, testset)
    evals = evals.replace(TRADUCTION)
    evals = evals.rename(columns=TRADUCTION_COLUMN)
    print(evals.groupby([granularity_str, 'Classifieur', 'Tâche']).mean().unstack().to_latex(
        float_format="{:,.2f}".format))
    #evals['classifier'] = evals.classifier.str.normalize('NFC')
    g = sns.factorplot(x=granularity_str, hue='Classifieur', y='mcc', data=evals, kind='violin',
                       inner='point', cut=0.1,
                       aspect=2, )
    g.set(ylim=(-0.7, 1.1))
    plt.savefig('/Users/olivier/Documents/ecole/maitrise/memoire/figures/results.pdf')

    df = evals[(evals[granularity_str] == 'Tout') | (evals[granularity_str] == 'Participant') | (
                evals[granularity_str] == 'Tâche')]
    df = df[df['Classifieur'] == TRADUCTION['randomforest']]
    g = sns.factorplot(x='participant', y='mcc', row=granularity_str, data=df, kind='box', aspect=1.75)
    g.set(ylim=(-0.7, 1.1))
    plt.savefig('/Users/olivier/Documents/ecole/maitrise/memoire/figures/results_participants.pdf')

    sns.factorplot(x='Tâche', y='mcc', col=granularity_str, data=df, kind='box', col_wrap=2, legend=True,
                   legend_out=True)
    plt.savefig('/Users/olivier/Documents/ecole/maitrise/memoire/figures/results_taskname.pdf')

    # g = sns.factorplot(x='participant', y='mcc', data=df, kind='violin', inner='point', cut=0.1, aspect=2)
    # g.set(ylim=(-0.7, 1.1))
    # plt.savefig('/Users/olivier/Documents/ecole/maitrise/memoire/figures/results_participant.pdf')


def eval_violin(granularities, info, model_types, testset):
    evals = []
    for granularity in granularities:
        model_dir = os.path.join('basic_models', granularity)
        for classifier in model_types:
            models = Models(model_dir, classifier)
            try:
                eval = evaluate_classfier(testset, info, models, context=False, groupby=['participant', 'taskname'])
                # eval.loc[:, ['classifier']] = classifier
                eval.name = 'mcc'
                eval = eval.reset_index()
                eval['classifier'] = classifier
                eval['granularity'] = granularity
                evals.append(eval)
            except KeyError:
                pass
    evals = pandas.concat(evals, ignore_index=True)
    with open('evals_violin.pkl', 'wb') as f:
        pickle.dump(evals, f)
    return evals


def mean_basic(testset, info):
    granularities = ['taskname', 'participant', 'all', 'generalize_participant', 'generalize_taskname']
    model_types = ['randomforest', 'svm', 'knn']  #, 'freq']
    evals = []
    if os.path.exists('evals_mean.pkl'):
        with open('evals_mean.pkl', 'rb') as f:
            evals = pickle.load(f)
    else:
        evals = eval_mean(evals, granularities, info, model_types, testset)

    evals = evals.replace(TRADUCTION)
    evals = evals.rename(columns=TRADUCTION_COLUMN)
    print(evals.to_latex(float_format="{:,.2f}".format, index=None))
    g = sns.factorplot(x=TRADUCTION_COLUMN['granularity'], hue='Classifieur', y='mcc', data=evals, kind='bar', aspect=2,
                       legend_out=True)
    g.set(ylim=(-0.7, 1.1))
    plt.savefig('/Users/olivier/Documents/ecole/maitrise/memoire/figures/results_mean.pdf')


def eval_mean(evals, granularities, info, model_types, testset):
    for granularity in granularities:
        for classifier in model_types:
            eval = {}
            model_dir = os.path.join('basic_models', granularity)
            models = Models(model_dir, classifier)
            try:
                eval['mcc'] = evaluate_classfier(testset, info, models, groupby=None, context=False)
                eval['classifier'] = classifier
                eval['granularity'] = granularity
                evals.append(eval)
            except KeyError:
                pass
    evals = pandas.DataFrame(evals)
    with open('evals_mean.pkl', 'wb') as f:
        pickle.dump(evals, f)
    return evals


def context_compare(testset, info):
    granularities = ['Tout', 'Tâche', 'Participant', 'generalize_participant', 'generalize_taskname']
    model_types = ['randomforest', 'knn', 'svm']
    context = [False, True]
    if os.path.exists('evals_context.pkl'):
        with open('evals_context.pkl', 'rb') as f:
            evals = pickle.load(f)
    else:
        evals = eval_context(context, granularities, info, model_types, testset)
        with open('evals_context.pkl', 'wb') as f:
            pickle.dump(evals, f)
    evals = evals.replace(TRADUCTION)
    evals = evals.rename(columns=TRADUCTION_COLUMN)
    g = sns.factorplot(x='Classifieur', hue='context', y='mcc', col=granularity_str, data=evals, kind='box', col_wrap=2,
                       col_order=granularities[:3], aspect=1)
    plt.savefig('/Users/olivier/Documents/ecole/maitrise/memoire/figures/results_context.pdf')


def eval_context(context, granularities, info, model_types, testset):
    evals = []
    # for granularity in granularities:
    #    for classifier in model_types:
    for granularity, classifier, context in itertools.product(granularities, model_types, context):
        if context:
            model_dir = os.path.join('context', granularity)
        else:
            model_dir = os.path.join('basic_models', granularity)
        models = Models(model_dir, classifier)
        try:
            eval = evaluate_classfier(testset, info, models, context=context, groupby=['participant', 'taskname'])
            # eval.loc[:, ['classifier']] = classifier
            eval.name = 'mcc'
            eval = eval.reset_index()
            eval['classifier'] = classifier
            eval['granularity'] = granularity
            eval['context'] = context
            evals.append(eval)
        except KeyError:
            pass
    evals = pandas.concat(evals, ignore_index=True)
    return evals


if __name__ == "__main__":
    dataset_path = 'datasets_begin'
    with open(os.path.join(dataset_path, 'info.json')) as f:
        info = json.load(f)
    evals = []
    testset = pandas.read_csv(os.path.join(dataset_path, 'testset.csv'), index_col='Time', parse_dates=True).dropna()
    mean_basic(testset, info)
    violin_basic(testset, info)
    context_compare(testset, info)
    plt.show()
