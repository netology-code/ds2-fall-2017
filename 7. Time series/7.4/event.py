__author__ = 'artonson'

from datetime import timedelta
from itertools import izip

import numpy as np

class ChangepointEvent(object):
    def __init__(self, ts, params, delay=None):
        self.ts = ts
        self.params = params
        self.delay = delay
        self.params_before = None

    def __iter__(self):
        yield self.ts
        yield self.params

    def initialise(self, ts_begin):
        if None is self.ts:
            self.ts = ts_begin + self.delay

    def next_subevent(self, ts, sampler_params):
        assert self.ts is not None, 'changepoint occurrence time is undefined'
        if ts >= self.ts:
            return None         # self corresponds to some past event
        self.params_before = sampler_params
        return self
