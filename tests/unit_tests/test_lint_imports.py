# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import pytest
import subprocess
import sys

def test_import_linter():
    """Test that import graph contracts are satisfied"""
    # Install import-linter if not available
    try:
        import importlinter  # noqa
    except ImportError, ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "import-linter"])

    # Run the lint-imports command
    result = subprocess.run(
        ["lint-imports", "--debug", "--verbose", "--no-cache"],
        cwd=".",  # Run from project root
        capture_output=True,
        text=True
    )
    print(result.stdout)

    assert result.returncode == 0, (
        f"Import linting failed (exit code {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
