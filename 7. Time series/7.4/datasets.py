from datetime import datetime
import json
from itertools import izip

from datatypes import DataPoint
from event import ChangepointEvent


DATE_FORMAT = '%d.%m.%Y %H:%M:%S.%f'


def read_dataset(data_file, changepoints_file=None):
    data, changepoints = [], []
    data_length = int(data_file.readline().strip())
    for time_series_index in xrange(data_length):
        data.append(read_dataset_once(data_file))
    if None is not changepoints_file:
        data_length = int(changepoints_file.readline().strip())
        for cp_list_index in xrange(data_length):
            cp_list_length = int(changepoints_file.readline().strip())
            for cp_index in xrange(cp_list_length):
                ts, params = changepoints_file.readline().strip().split('\t')
                ts = datetime.strptime(ts, DATE_FORMAT)
                changepoints.append(ChangepointEvent(ts, json.loads(params)))
    return data, changepoints


def read_dataset_once(data_file):
    time_series_length = int(data_file.readline().strip())
    time_series = []
    for point_index in xrange(time_series_length):
        ts, value = data_file.readline().strip().split('\t')
        ts = datetime.strptime(ts, DATE_FORMAT)
        time_series.append(DataPoint(ts, float(value)))
    return time_series


def write_dataset(data_file, changepoints_file, dataset_length, data, changepoints):
    data_file.write('{}\n'.format(dataset_length))
    for time_series in data:
        write_dataset_once(data_file, time_series)
    changepoints_file.write('{}\n'.format(dataset_length))
    for cp_list in changepoints:
        changepoints_file.write('{}\n'.format(len(cp_list)))
        for ts, params in cp_list:
            changepoints_file.write('{ts}\t{params}\n'.format(
                ts=ts.strftime(DATE_FORMAT), params=json.dumps(params)))


def write_dataset_once(data_file, time_series):
    data_file.write('{}\n'.format(len(time_series)))
    for ts, value in time_series:
        data_file.write('{ts}\t{value}\n'.format(
            ts=ts.strftime(DATE_FORMAT), value=value))


def write_simple_dataset(data_file, dataset_length, data, changepoints):
    data_file.write('{}\n'.format(dataset_length))
    for time_series, cp_list in izip(data, changepoints):
        write_simple_dataset_once(data_file, time_series, cp_list)


def write_simple_dataset_once(data_file, time_series, cp_list):
    data_file.write('{}\n'.format(len(time_series)))
    is_changepoint = False
    iter_changepoints = iter(cp_list)
    changepoint = next(iter_changepoints, None)
    for ts, value in time_series:
        if None is not changepoint and ts > changepoint.ts:
            is_changepoint = not is_changepoint
            changepoint = next(iter_changepoints, None)
        data_file.write('{ts}\t{value}\t{is_changepoint}\n'.format(
            ts=ts.strftime(DATE_FORMAT), value=value, is_changepoint=int(is_changepoint)))


def read_simple_dataset(data_file, keep_target=False):
    data_points, changepoints, targets = [], [], []
    data_length = int(data_file.readline().strip())
    for iteration in xrange(data_length):
        results = read_simple_dataset_once(data_file, keep_target=keep_target)
        data_points.append(results[0])
        changepoints.append(results[1])
        if keep_target:
            targets.append(results[2])
    if keep_target:
        return data_points, changepoints, targets
    else:
        return data_points, changepoints


def read_simple_dataset_once(data_file, keep_target=False, data_length=None):
    time_series, cp_list, target = [], [], []
    if None is data_length:
        time_series_length = int(data_file.readline().strip())
    else:
        time_series_length = data_length
    is_changepoint = False
    for point_index in xrange(time_series_length):
        ts, value, is_current_changepoint = data_file.readline().strip().split('\t')
        is_current_changepoint = int(is_current_changepoint)
        ts = datetime.strptime(ts, DATE_FORMAT)
        time_series.append(DataPoint(ts, float(value)))
        if is_changepoint != is_current_changepoint:
            cp_list.append(ChangepointEvent(ts, None))
        is_changepoint = is_current_changepoint
        target.append(int(is_current_changepoint))
    if keep_target:
        return time_series, cp_list, target
    else:
        return time_series, cp_list


def save_simple_format(data, lengths, prefix):
    train_data, train_changepoints, test_data, test_changepoints = data
    train_length, test_length = lengths
    with open(prefix + '.train_data', 'w') as data_file:
        write_simple_dataset(data_file,
                             train_length,
                             train_data,
                             train_changepoints)
    with open(prefix + '.test_data', 'w') as data_file:
        write_simple_dataset(data_file,
                             test_length,
                             test_data,
                             test_changepoints)


def save_verbose_format(data, lengths, prefix):
    train_data, train_changepoints, test_data, test_changepoints = data
    train_length, test_length = lengths
    with open(prefix + '.train_data', 'w') as data_file:
        with open(prefix + '.train_changepoints', 'w') as changepoints_file:
            write_dataset(data_file,
                          changepoints_file,
                          train_length,
                          train_data,
                          train_changepoints)
    with open(prefix + '.test_data', 'w') as data_file:
        with open(prefix + '.test_changepoints', 'w') as changepoints_file:
            write_dataset(data_file,
                          changepoints_file,
                          test_length,
                          test_data,
                          test_changepoints)
