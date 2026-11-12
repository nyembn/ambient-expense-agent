# Copyright 2026 Google LLC
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

"""Configuration for the ambient expense approval agent."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Agent runtime configuration."""

    review_threshold: float = float(os.getenv("EXPENSE_REVIEW_THRESHOLD", "100.0"))
    model: str = os.getenv("EXPENSE_MODEL", "gemini-3.1-flash-lite")


config = Config()
