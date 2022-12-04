
# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Fionn Malone <fmalone@google.com>
#

import importlib
import sys

def purge_ipie_modules():
    modules = [m for m in sys.modules.keys() if 'ipie' in m]
    for m in modules:
        del sys.modules[m]

class Config:

    def __init__(self):
        self.options = {}

    def add_option(self, key, val):
        self.options[key] = val

    def update_option(self, key, val):
        _val = self.options.get(key)
        if _val is None:
            raise KeyError("config option not found: {}".format(_val))
        self.options[key] = val

    def get_option(self, key):
        _val = self.options.get(key)
        if _val is None:
            raise KeyError("config option not found: {}".format(_val))
        return _val

    def __str__(self):
        _str = ''
        for k, v in self.options.items():
            _str += '{} : {}\n'.format(k, v)
        return _str

config = Config()

# FDM: Hack to cope with pytest preemtively importing the whole module in search
# for tests thus breaking assumption config is only set once per session.
# Otherwise I couldnt figure out a way to successfully reload ipie.utils.backend
# in order for arraylib etc to be correctly set.
import os
IPIE_USE_GPU = os.environ.get('IPIE_USE_GPU', False)
# Default to not using for the moment.
config.add_option('use_gpu', bool(int(IPIE_USE_GPU)))
# Memory limits should be in GB
config.add_option('max_memory_for_wicks', 2.0)
config.add_option('max_memory_sd_energy_gpu', 2.0)
