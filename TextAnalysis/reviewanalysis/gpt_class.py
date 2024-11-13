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
from tqdm import tqdm

from openai import OpenAI, OpenAIError
    
OPEN_AI_CACHE = jp(os.path.dirname(os.path.dirname(__file__)), '..', 'openaicache')

# API_KEY_FILE = jp(HOME_DIR, 'credentials', 'openai_pmolnar_gsu_edu.apikey')
API_KEY_FILE = '/opt/truist_workshop_openai.apikey'


DEFAULT_PROMPT = """Analyze the provided review delimited by <> and extract product-related
aspects, sentiments (positive/negative/neutral), and justifications.
Format the result as JSON object with the following structure: a key named 'aspect'
containing a nested object with keys for each aspect, and each aspect having an object with
'sentiment' and 'justification' keys.
Exclude non-product discussions. If no sentiment is expressed for an aspect, return
'neutral' as the sentiment and 'No information provided' as the value. If information
can't be retrieved, return an empty JSON object.
Review: <{review}>
"""

CATEGORY_PROMPT = """
Define a category for the list of key terms delimited by <>. Only output the category as JSON object. The list of key terms is: <{key_terms}>
"""


class GPT:
    
    def __init__(self, api_key: str = '', open_ai_cache = OPEN_AI_CACHE, create_cache_dir: bool = False):
        if len(api_key) > 0:
            self.api_key = api_key
        else:
            self.api_key = open(API_KEY_FILE, 'r', encoding='utf-8').read().strip()
            
        self.open_ai_cache = open_ai_cache
        if not os.path.isdir(create_cache_dir):
            os.makedirs(self.open_ai_cache, exist_ok = True)
        assert os.path.isdir(self.open_ai_cache), f"Cache directory does not exist: {self.open_ai_cache}"
        
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key = self.api_key
        )
        
        self.prompts = { 'default': DEFAULT_PROMPT }
        
        
    def list_models(self) -> List[str]:
        models = self.client.models.list()
        return [ model.id for model in models.data ]
    
    def __embedd_or_cache(self, text: str, model: str = 'text-embedding-ada-002'):
        hexhash = hashlib.sha3_224(text.encode('UTF-8')).hexdigest()
        cache_dir = jp(self.open_ai_cache, model)
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        file_name = jp(cache_dir, f"{hexhash}.json")
        
        if os.path.isfile(file_name):
            logging.debug(f"Used cached response from file: {file_name}")
            content = open(file_name, "r", encoding='utf-8').read()
            content_data = json.loads(content)
            content_data['cached_query'] = True
            
        else:
            logging.debug("Embedding with OpenAI...")
            T_0 = T_now()
            
            response = self.client.embeddings.create(input=text, model=model)
            try:
                logging.warning(response.warning)
            except:
                pass
            
            content_json = response.model_dump_json()
            content_data = json.loads(content_json)
            T_delta = T_now()-T_0
            elapsed_seconds = T_delta.total_seconds()
            content_data['meta'] = {
                'elapsed_seconds': elapsed_seconds,
                'hash': hexhash,
            }
            logging.debug(f"done. Elapsed time: {T_delta}")
            with open(file_name, "w", encoding='utf-8') as io:
                json.dump(content_data, io)
            content_data['cached_query'] = False
            
        return content_data
                         
    
        
    def embedding(self, text: str, model: str = 'text-embedding-ada-002') -> np.array:
        embedding_struct = self.__embedd_or_cache(text, model = model)
        vect = np.array(embedding_struct['data'][0]['embedding'])
        return vect
                         
    
    def embedding_from_table(self, sentiment_df: pd.DataFrame, vect_col: str = None,
                               model: str = 'text-embedding-ada-002',
                               text_col: str = 'aspect', product_number_col: str = 'Product Model Number',
                               id_col: str = 'ID', group_id_col: str = 'Duplicate Group',
                               rating_col: str = 'Star Rating') -> pd.DataFrame:
        
        aspect_df = sentiment_df[text_col].drop_duplicates().reset_index()
        
        T_0 = T_now()
        aspect_df[vect_col if vect_col else model] \
            = aspect_df[text_col].map(lambda txt: self.embedding(txt, model=model), na_action='ignore')
        
        aspect_df.drop('index', axis=1, inplace=True)
        logging.info(f"\n\nDone. Elapsed time: {T_now()-T_0}")
        
        embedding_df = pd.merge(sentiment_df, aspect_df, on = text_col, how = 'left')
        logging.info(f"Number of records: {embedding_df.shape[0]:,}, number of fields: {embedding_df.shape[1]:,}")
        
        return embedding_df
                                         
    
    def __create_or_cache(self, prompt: str, prompt_label: str = 'default'):
        hexhash = hashlib.sha3_224(prompt.encode('UTF-8')).hexdigest()
        cache_dir = jp(self.open_ai_cache, prompt_label)
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        file_name = jp(cache_dir, f"{hexhash}.json")
        if os.path.isfile(file_name):
            logging.debug(f"Used cached response from file: {file_name}")
            content = open(file_name, "r", encoding='utf-8').read()
            contdata = json.loads(content)
            contdata['cached_query'] = True
            
        else:
            logging.debug("Query OpenAI...")
            T_0 = T_now()
            response = self.client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages = [
                    {"role": "user", "content": prompt}
                ],
                temperature = 0,
            )
            
            try:
                logging.warning(response.warning)
            except:
                pass
            
            content = response.choices[0].message.content
            contdata = json.loads(content)
            T_delta = T_now()-T_0
            elapsed_seconds = T_delta.total_seconds()
            contdata['meta'] = {
                'elapsed_seconds': elapsed_seconds,
                'hash': hexhash,
                'prompt_label': prompt_label,
            }
            logging.debug(f"done. Elapsed time: {T_delta}")
            with open(file_name, "w", encoding='utf-8') as io:
                json.dump(contdata, io)
            contdata['cached_query'] = False
            
        return contdata
            
    def query(self, prompt: str, prompt_label: str = 'default'):
        return self.__create_or_cache(prompt, prompt_label)
    
    def analyze_review(self, review: str, prompt_label:str = 'default') -> pd.DataFrame:
#         prompt = f"""Analyze the provided review delimited by <> and extract product-related
# aspects, sentiments (positive/negative/neutral), and justifications.
# Format the result as JSON object with the following structure: a key named 'aspect'
# containing a nested object with keys for each aspect, and each aspect having an object with
# 'sentiment' and 'justification' keys.
# Exclude non-product discussions. If no sentiment is expressed for an aspect, return
# 'neutral' as the sentiment and 'No information provided' as the value. If information
# can't be retrieved, return an empty JSON object.
# Review: <{review}>
# """
        prompt = self.prompts[prompt_label].format(review = review)
        resp = self.__create_or_cache(prompt, prompt_label=prompt_label)
        if len(resp.get('aspect', [])) > 0:
            data = [{
                    'aspect': asp,
                    'sentiment': val['sentiment'],
                    'justification': val['justification'],
                    'elapsed_seconds': resp.get('meta', {}).get('elapsed_seconds'),
                    'hash': resp.get('meta', {}).get('hash'),
                    'prompt_label': resp.get('meta', {}).get('prompt_label'),
                    }
                     for asp, val in resp['aspect'].items()
                ]
            return pd.DataFrame.from_records(data)
        else:
               return pd.DataFrame({
                    'aspect': [],
                    'sentiment': [],
                    'justification': [],
                    'elapsed_seconds': [],
                    'hash': [],
                    'prompt_label': [],
               })
               
    
    
    def category_for_key_terms(self, key_terms: Union[str, List], prompt_label = 'category') -> str:
        if isinstance(key_terms, str):
            prompt = CATEGORY_PROMPT.format(key_terms = key_terms)
        else:
            prompt = CATEGORY_PROMPT.format(key_terms = ', '.join(key_terms))
        resp = self.__create_or_cache(prompt, prompt_label=prompt_label)
        return resp
    
    
    
    def list_cache_files(self, label: str = 'default') -> List[str]:
        return glob.glob(jp(self.open_ai_cache, label, "*"))
    
    
    
    def analyze_reviews_from_table(self, product_df: pd.DataFrame, prompt_label: str = 'default', limit: int = 0, to_lower=False,
                                   text_col: str = 'Review Text', product_number_col: str = 'Product Model Number',
                                   id_col: str = 'ID', group_id_col: str = 'Duplicate Group',
                                   rating_col: str = 'Star Rating') -> pd.DataFrame:
        sentiment_df = None
        T_0 = T_now()
        n_limited = product_df.shape[0] if limit<=0 else min(product_df.shape[0], limit)
        logging.info(f"ANALYZE REVIEWS ({n_limited:,} records)")
        
        for j, (ind, row) in tqdm(enumerate(product_df.iterrows()), desc="Processing...", ascii=False, ncols=75):
            if (limit>0) and (j>limit):
                break
            if pd.notnull(row[text_col]):
                review = row[text_col]
                review_id = row[id_col]
                review_duplicate_group = row[group_id_col]
                review_star_rating = row[rating_col]
                logging.debug(f"{review_id} ({review_star_rating}): {review[:70]}")

                analysis_df = self.analyze_review(review)
                for col in [product_number_col, id_col, group_id_col, rating_col]:
                    analysis_df[col] = row[col]

                if sentiment_df is None:
                    sentiment_df = analysis_df
                else:
                    sentiment_df = pd.concat([sentiment_df, analysis_df], axis=0)

        if to_lower:
            sentiment_df['aspect'] = sentiment_df.aspect.map(lambda s: s.lower().strip(), na_action='ignore')
        logging.info(f"\n\nDone. Elapsed time: {T_now()-T_0}")
        logging.info(f"Number of records: {sentiment_df.shape[0]:,}, number of fields: {sentiment_df.shape[1]:,}")
        return sentiment_df