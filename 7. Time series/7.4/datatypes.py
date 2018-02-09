from collections import namedtuple
from itertools import izip
from operator import attrgetter

__all__ = ["DataPoint", "ParamValue", "ParamData", "TsData", "ParamTsValue"]

DataPoint = namedtuple("DataPoint", "ts, value")
ParamValue = namedtuple("ParamValue", "param, value")
ParamData = namedtuple("ParamData", "param, data")
TsData = namedtuple("TsData", "ts, data")
ParamTsValue = namedtuple("ParamTsValue", "param, ts, value")

ts_from_point = attrgetter('ts')
val_from_point = attrgetter('value')


def ts_from_data(data):
    return map(ts_from_point, data)


def val_from_data(data):
    return map(val_from_point, data)


def split_data(data):
    return ts_from_data(data), val_from_data(data)


def zip_data(timestamps, values):
    return [DataPoint(ts, value) for ts, value in izip(timestamps, values)]
