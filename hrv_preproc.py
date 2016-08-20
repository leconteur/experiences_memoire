from os import path
from glob import glob
import subprocess
import shlex
import pandas as pd


def process_session(basedir):
    ecg_filename = glob(path.join(basedir, '*ECG.csv'))[0]
    temp_output = path.basename(ecg_filename[:-4])
    participant, session = basedir.split('/')[-3:-1]

    wrsamp(ecg_filename, temp_output)
    wqrs(temp_output)
    ann2rr(ecg_filename, temp_output)
    with open(ecg_filename[:-4] + '_RR.beats') as f:
        times = f.readlines()
    times = (pd.DataFrame({'RtoR':times}).astype(float) * 1000).astype(int)
    times['Time'] = None
    outpath = path.join(basedir, 'RR_local.csv')
    times.to_csv(outpath, header=True, index=False, columns=['Time', 'RtoR'])
    # Rscript hrv.r matin p01
    command = 'Rscript hrv.r {session} {participant}'

    command = shlex.split(command.format(session = session, participant=participant))
    ret = subprocess.call(command)
    if ret != 0:
        raise Exception('Something went wrong in the R script')


def ann2rr(ecg_filename, temp_output):
    # ann2rr -r ecg02 -a wqrs -i s > test.txt
    command = shlex.split('ann2rr -r {annfile} -a wqrs -i s'.format(annfile=temp_output))
    with open(ecg_filename[:-4] + '_RR.beats', 'w') as f:
        ret = subprocess.call(command, stdout=f)
        if ret != 0:
            raise Exception('Something went wrong in ann2rr')


def wqrs(temp_output):
    command_wqrs = shlex.split('wqrs -r {} -v'.format(temp_output))
    ret = subprocess.call(command_wqrs)
    if ret != 0:
        raise Exception('Something went wrong in wqrs')


def wrsamp(ecg_filename, temp_output):
    # wrsamp -F 250 -G 1 -f 1 -i 2015_08_05-08_26_01_ECG.csv -s "," -o ecg02 1
    command_string = 'wrsamp -F {frequency} -G {gain} -f {skiplines} -i {input_file} -s "," -o {outputfile} 1'
    command = command_string.format(frequency=250, gain=1, skiplines=1, input_file=ecg_filename,
                                    outputfile=temp_output)
    command = shlex.split(command)
    ret = subprocess.call(command)
    if ret != 0:
        raise Exception('Something went wrong in wrsamp')


if __name__ == "__main__":
    basedir = '/Users/olivier/Documents/ecole/maitrise/rawdata/*/*/bioharness'
    folders = glob(basedir)

    for f in folders:
        process_session(f)
