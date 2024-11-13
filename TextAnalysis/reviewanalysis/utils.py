from typing import List, Dict, Any, Union
import logging
import os
import sys
import pandas as pd


def print_shape(df: pd.DataFrame):
    msg = f"Number of records: {df.shape[0]:,}, number of fields: {df.shape[1]:,}"
    logging.info(msg)
