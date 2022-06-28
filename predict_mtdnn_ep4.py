#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 10:43:49 2022

@author: dimakarp1996
"""

import shutil
import os
import sys, gzip
import json
from zipfile import ZipFile
import pandas as pd
os.chdir('/cephfs/home/karpov.d/mt-dnn/checkpoints')


class Task:
    def __init__(self, name, filename, default_batch, classes, task_id):
        self.name = name
        self.filename = filename
        self.default_batch = default_batch
        self.classes = classes
        self.task_id = task_id


default_sst_batch = [(-1, 'a')]
default_rte_batch = [(-1, ('a', 'a'))]
default_copa_batch = [(-1, ('a', ['a', 'a']))]
default_record_batch = [(-1, (1, 'a', 'a', 'a', 444))]

tasks = [Task('cola', 'CoLA.tsv', default_sst_batch, [0, 1], 0),
         Task('sst2', 'SST-2.tsv', default_sst_batch, [0, 1], 1),
         Task('qqp', 'QQP.tsv', default_rte_batch, [0, 1], 2),
         Task('mrpc', 'MRPC.tsv', default_rte_batch, [0, 1], 3),
         Task('rte', 'RTE.tsv', default_rte_batch, ['not_entailment', 'entailment'], 4),
         Task('mnli-m', 'MNLI-m.tsv', default_rte_batch, ['contradiction', 'neutral','entailment'], 5),
         Task('mnli-mm', 'MNLI-mm.tsv', default_rte_batch, ['contradiction', 'neutral','entailment'], 5),
         Task('qnli', 'QNLI.tsv', default_rte_batch, ['not_entailment', 'entailment'], 6),
         Task('stsb', 'STS-B.tsv', default_rte_batch, [1], 7),
         Task('ax', 'AX.tsv', default_rte_batch, ['entailment', 'not_entailment'], 4)]
def process(directory):
    FILES = [j  for j in os.listdir(os.getcwd()+'/'+directory) if 'epoch_4.json' in j and 'test' in j]
    shutil.copy2('/cephfs/home/karpov.d/mt-dnn/default_preds/AX.tsv',os.getcwd()+'/'+directory+'/AX.tsv')
    shutil.copy2('/cephfs/home/karpov.d/mt-dnn/default_preds/WNLI.tsv',os.getcwd()+'/'+directory+'/WNLI.tsv')    
    from collections import defaultdict
    print(f'Processing {directory}')
    for filename in FILES:
        print(f'Processing {filename}')
        dictionaries=defaultdict(lambda:defaultdict(int))
        json_file = json.load(open(os.getcwd()+'/'+directory+'/'+filename,'r'))
        name = filename.replace('_matched','-m').replace('_mismatched','-mm').split('_')[0].upper().replace('COLA','CoLA').replace('-MM','-mm').replace('-M','-m').replace('STSB','STS-B').replace('SST','SST-2')+'.tsv'
        our_task= [task for task in tasks if task.filename == name][0]
        if name=='STS-B.tsv':
            pred = [max(0, min(s,5)) for s in json_file['scores']]
        else:
            pred = [our_task.classes[s] for s in json_file['predictions']]
        #old_pred = pd.read_csv('old_pred/'+name,sep='\t')['prediction']
        #for true_,old_ in zip(pred,old_pred):
        #    dictionaries[true_][old_]+=1
        #print(name)
        #if 'STS' not in name:
        #    print(dictionaries)
        #breakpoint()
        pd.DataFrame({'prediction': pred}).to_csv(os.getcwd()+'/'+f'{directory}/'+name,sep='\t')
    print(f'Writing submit')
    to_submit = os.listdir(os.getcwd()+'/'+directory)
    if len(to_submit) >=10:
        zip_obj = ZipFile(os.getcwd()+'/'+f'submit_{directory}.zip','w')
        for f in to_submit:
            zip_obj.write(os.getcwd()+'/'+directory+'/'+f)
        zip_obj.close()
for directory in [k for k in os.listdir(os.getcwd()) if '2022-06' in k]:
    process(directory)
    # Reads the file by chunks to avoid exhausting memory
