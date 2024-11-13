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


from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class Embedding:
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        
    def embedding(self, text: str):
        return self.model.encode(text)
    
    
    def embedding_from_table(self, sentiment_df: pd.DataFrame, vect_col: str = None,
                               text_col: str = 'aspect', product_number_col: str = 'Product Model Number',
                               id_col: str = 'ID', group_id_col: str = 'Duplicate Group',
                               rating_col: str = 'Star Rating') -> pd.DataFrame:
        embedding_df = sentiment_df.copy()
        T_0 = T_now()
        embedding_df[vect_col if vect_col else self.model_name] \
            = embedding_df[text_col].map(self.embedding)
        
        logging.info(f"\n\nDone. Elapsed time: {T_now()-T_0}")
        logging.info(f"Number of records: {embedding_df.shape[0]:,}, number of fields: {embedding_df.shape[1]:,}")
        return embedding_df
    
