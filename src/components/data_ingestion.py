import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.constant import *
from src.exception import customException
from src.logger import logging

