#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
from fastai.text import *

#this finds our json files
path_to_json = 'data/docs'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

#creating and naming our dataframes columns
jsons_data = pd.DataFrame(columns=['job description', 'job id'])

#extracts necessary info from json files and stores them in the df 
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)

        job_description = json_text['jd_information']['description']
        job_id = json_text['_id']
        jsons_data.loc[index] = [job_description, job_id]

#reading and creating dept dataframe
csv_data = pd.read_csv("./data/document_departments.csv")
jsons_data.columns = ['Job Description', 'Document ID']

#merging both dataframes
jsons_data['Document ID']=jsons_data['Document ID'].astype(int)
final = jsons_data.merge(csv_data, on='Document ID')
final.drop('Document ID', axis=1, inplace=True)

#splitting dataframes in training and testing 
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(final, test_size=0.2)

cwd = os.getcwd()

#create language model data
data_lm = TextLMDataBunch.from_df(path=cwd, train_df=train_df, valid_df=valid_df, text_cols="Job Description", label_cols="Department")

#create classifier model data
data_clas = TextClasDataBunch.from_df(path=cwd, train_df=train_df, valid_df=valid_df, text_cols="Job Description", label_cols="Department", vocab=data_lm.train_ds.vocab, bs=16)

#fitting our model 
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.fit_one_cycle(1, 1e-2)
learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)
learn.save_encoder('ft_enc')
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')
learn.fit_one_cycle(1, 1e-2)
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
learn.fit_one_cycle(1, 1e-3)
