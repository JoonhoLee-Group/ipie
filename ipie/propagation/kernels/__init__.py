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
# Author: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#

# at config level import appropriate kernels.
from ipie.config import config

if config.get_option("use_gpu"):
    from .gpu.vhs import call_kernel_VHS_construction1
    from .gpu.vhs import call_kernel_VHS_construction2
else:
    call_kernel_VHS_construction1 = None
    call_kernel_VHS_construction2 = None

