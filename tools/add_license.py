NOTICE = """
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
"""

from pathlib import Path

from pathlib import Path
import subprocess


def get_authors(file_path):
    authors = subprocess.check_output(f"git shortlog {file_path} -n -s --email".split())
    first_names = [name.decode("utf_8") for name in authors.split()[1::4]]
    last_names = [name.decode("utf_8") for name in authors.split()[2::4]]
    email = [name.decode("utf_8") for name in authors.split()[3::4]]
    return list(zip(first_names, last_names, email))


files = [("bin/ipie", get_authors("bin/ipie"))]
for path in Path("bin").rglob("*.py"):
    files.append((path, get_authors(path)))
for path in Path("examples").rglob("*.py"):
    files.append((path, get_authors(path)))
for path in Path("ipie").rglob("*.py"):
    files.append((path, get_authors(path)))
for path in Path("lib").rglob("*.c"):
    files.append((path, get_authors(path)))
for path in Path("lib").rglob("*.h"):
    files.append((path, get_authors(path)))
for path in Path("lib").rglob("*.py"):
    files.append((path, get_authors(path)))

for file_path, names in files:
    with open(file_path, "r+") as file:
        file_data = file.read()
        # use for loop to avoid duplicates
        if "_version" in str(file_path):
            continue
        if "legacy" in str(file_path):
            continue
        if len(names) > 0:
            num_authors = 0
            string = ""
            for f, l, e in names:
                if f in string:
                    continue
                if ".com" not in e:
                    _email = ""
                    string += f"{f:s} {l:s}\n# {'':9s}"
                else:
                    _email = e
                    string += f"{f:s} {l:s} {_email:s}\n# {'':9s}"
                num_authors += 1
            authors = f"""#
# Author{('s' if num_authors > 1 else ''):s}: {string.strip()}
"""
        file.seek(0, 0)
        file.write(NOTICE + authors + "\n" + file_data)
