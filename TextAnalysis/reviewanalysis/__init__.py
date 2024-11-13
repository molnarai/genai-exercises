from typing import List, Dict, Tuple, Union, Any
import sys
import os
jp = os.path.join
import datetime
T_now = datetime.datetime.now
import json
import re
import glob
import numpy as np
import pandas as pd
import hashlib
import logging

from openai import OpenAI, OpenAIError


#
# Definitions (system specific)
#



#
# Import
#

from .utils import (
    print_shape
)

#
# Import classes
#
from .gpt_class import GPT
from .text_embedding_class import Embedding

#
# Other methods
#
    




# def print_shape(df: pd.DataFrame):
#     print(f"Number of records: {df.shape[0]:,}, number of fields: {df.shape[1]:,}")

    
def load_customer_reviews(reviews_path: str = REVIEWS_PATH,
                          reviews_xlsx: str = CURRENT_REVIEWS_XLSX) -> pd.DataFrame:
    df = pd.read_excel(jp(reviews_path, reviews_xlsx))
    df['ID'] = range(1, df.shape[0]+1)
    print_shape(df)
    return df


def filter_by_styles(df, sel_styles = SELECTED_STYLES_STR, unique: bool = False) -> pd.DataFrame:
    mask1 = df['Product Model Number'].map(lambda pmn: pmn  in sel_styles)
    if unique:
        mask2 = df['Unique Review'] == "Unique"
        df2 = df[mask1&mask2]
    else:
        df2 = df[mask1]
    
    print_shape(df2)
    return df2
    
    
def create_product_dict(df: pd.DataFrame) -> Dict:
    return df[['Product Model Number', 'Product Title',
                'Manufacturer', 'Brand', 'Account Category',]] \
            .drop_duplicates() \
            .set_index('Product Model Number') \
            .to_dict(orient='index')



               
    
