from glob import glob
import os

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from performance import correct, resample
from skl.performance import load_eyetracker, contiguous_regions

plt.style.use('ggplot')

BASEDIR = "../../rawdata/"
clean_dir = "../../cleandata/"


def load_task_data(participant_name, session):
    filenames = glob(os.path.join(BASEDIR, participant_name, session, '*[hw].log'))
    # columns = ['participant response', 'target', 'reaction time', 'timestamp']
    data = [pd.read_csv(f, index_col='timestamp', parse_dates=True) for f in filenames]
    data = pd.concat(data)
    data['correct'] = data.apply(correct, axis=1)
    data = data[['correct', 'reaction time', 'workload', 'taskname']]
    data = resample(data, 'S', 1, 'bfill')
    return data


def load_bh(participant_name, session):
    filepath = os.path.join(BASEDIR, participant_name, session, 'bioharness', '*Summary.csv')
    filenames = glob(filepath)
    # na_values = {'HR': [0], 'BR': [6553.5], 'BRAmplitude': [0], 'ECGAmplitude': [0],
    #             'HRV': [65535], 'CoreTemp': [6553.5], 'Time': [], 'Posture': []}
    na_values = {'HR': [0], 'BR': [6553.5], 'BRAmplitude': [0], 'ECGAmplitude': [0],
                 'CoreTemp': [6553.5], 'Time': [], 'Posture': []}
    data = pd.read_csv(filenames[0], index_col='Time', parse_dates=True, dayfirst=True, usecols=na_values.keys(),
                       na_values=na_values)
    data = data.resample('S')
    if participant_name == 'p02':
        data.index += pd.DateOffset(hours=1)
    return data


def load_hrv(participant_name, session):
    filename = os.path.join(BASEDIR, participant_name, session, 'hrv.csv')
    data = pd.read_csv(filename, parse_dates=True, index_col='ts',
                       usecols=['HRV', 'ULF', 'VLF', 'LF', 'HF', 'LFHF', 'ts'])
    data = data.resample('S')
    if participant_name == 'p02':
        data.index += pd.DateOffset(hours=1)
    return data


def load_et(participant_name, session):
    filename = glob(os.path.join(BASEDIR, participant_name, session, 'eyetracker.txt'))[0]
    etdata = load_eyetracker(filename)
    etdata = etdata[['avg_x', 'avg_y', 'fix', 'lefteye_psize', 'righteye_psize']]
    etdata = compute_features(etdata)
    return etdata


def compute_features(etdata):
    distance_from_screen = 0.5  # meters
    pixels_linear_density = 2500.  # pixels per meters
    no_data = (etdata.avg_x == 0) & (etdata.avg_y == 0)
    idx = contiguous_regions(no_data.values, 3, 45)
    blink = pd.Series(np.zeros((len(etdata),)), index=etdata.index)
    blink.iloc[idx] = 1
    etdata.loc[no_data, :] = np.nan
    x_deg = np.arctan(distance_from_screen * pixels_linear_density / etdata.avg_x) * (180 / np.pi)
    y_deg = np.arctan(distance_from_screen * pixels_linear_density / etdata.avg_y) * (180 / np.pi)
    # velocity = (etdata[['avg_x', 'avg_y']].diff() ** 2).sum(axis=1) ** 0.5
    x_velocity = x_deg.diff()
    y_velocity = y_deg.diff()
    total_velocity = ((x_velocity ** 2) + (y_velocity ** 2)) ** 0.5
    saccade = pd.Series(np.zeros((len(etdata))), index=etdata.index)
    saccade.loc[total_velocity > 30] = 1
    idx = contiguous_regions((total_velocity < 30).values, 0, 3)
    saccade.iloc[idx] = 1
    involontary_fixation = pd.Series(np.zeros(len(etdata), ), index=etdata.index)
    idx = contiguous_regions((total_velocity < 30).values, 4, 8)
    involontary_fixation.iloc[idx] = 1

    fixation = pd.Series(np.zeros(len(etdata), ), index=etdata.index)
    fixation[(saccade == 0) & (involontary_fixation == 0)] = 1
    # velocity = np.arctan(distance_from_screen * pixels_linear_density / velocity)
    etdata['eye_velocity'] = total_velocity

    etdata = etdata.resample('s')

    blink = blink.resample('s', how='sum')
    etdata['blink_mean_60_seconds'] = pd.rolling_mean(blink, 60, 1, 's', center=False)
    etdata['blink_mean_10_seconds'] = pd.rolling_mean(blink, 10, 1, 's', center=False)

    saccade = saccade.resample('s')
    etdata['saccade_mean_60_seconds'] = pd.rolling_mean(saccade, 60, 1, 's', center=False)
    etdata['saccade_mean_10_seconds'] = pd.rolling_mean(saccade, 10, 1, 's', center=False)

    fixation = fixation.resample('s')
    etdata['fixation_mean_60_seconds'] = pd.rolling_mean(fixation, 60, 1, 's', center=False)
    etdata['fixation_mean_10_seconds'] = pd.rolling_mean(fixation, 10, 1, 's', center=False)

    involontary_fixation = involontary_fixation.resample('s')
    etdata['involontary_fixation_mean_60_seconds'] = pd.rolling_mean(involontary_fixation, 60, 1, 's', center=False)
    etdata['involontary_fixation_mean_10_seconds'] = pd.rolling_mean(involontary_fixation, 10, 1, 's', center=False)

    return etdata


def load_participant_data(participant_name):
    data = []
    for session in ['matin', 'soir']:
        print('    Converting session {}'.format(session))
        print('        Processing task')
        taskdata = load_task_data(participant_name, session)
        print('        Processing bioharness')
        bh_data = load_bh(participant_name, session)
        print('        Processing HRV')
        hrv_data = load_hrv(participant_name, session)
        print('        Processing eyetracker')
        et_data = load_et(participant_name, session)
        merged_data = pd.concat([taskdata, bh_data, hrv_data, et_data], axis=1)
        merged_data['timeofday'] = 'morning' if session == 'matin' else 'evening'
        data.append(merged_data)
    data = pd.concat(data)
    data['participant'] = participant_name
    return data


def make_dataset(participants):
    participants = ['p0{}'.format(i) for i in participants]
    data = []
    for p in participants:
        print('Converting participant {}'.format(p))
        data.append(load_participant_data(p))
    return pd.concat(data)


if __name__ == "__main__":
    participants = range(1, 9)
    data = make_dataset(participants)
    data.to_csv(os.path.join(clean_dir, 'test_2.csv'))
