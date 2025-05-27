import mne # type: ignore
import warnings
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from argparse import Namespace

from data.CHBMIT_old import Event, EDF
import random
import prettytable as pt

warnings.filterwarnings('ignore')
        
def infoParse(info):
        content_list = info.split('\n')
        try:
            content_list.remove('')
        except ValueError:
            pass
        file_info_list = content_list[0].split(': ')
        file_name = file_info_list[1]
        seizure_info_list = content_list[4:]
        seizure_list = []
        if seizure_info_list:
            for index in range(0, len(seizure_info_list), 2):
                try:
                    st_info_list = seizure_info_list[index].split(': ')
                    et_info_list = seizure_info_list[index+1].split(': ')
                    st = int((st_info_list[1].split(' s'))[0])
                    et = int((et_info_list[1].split(' s'))[0])
                    seizure_list.append(Event(st, et))
                except IndexError:
                    pass
        return EDF(file_name, seizure_list)
    
def parse(summary_path):
        edfs = []
        with open(summary_path) as f:
            content = f.read()
            info_list = content.split('\n\n')
            for info in info_list:
                if 'File Name' in info:
                    edfs.append(infoParse(info))
        return edfs


class CHBMIT(Dataset):
    
    
    # feat_cols = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 
    #              'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']
    feat_cols = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 
                 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10',
                 'FT10-T8', 'T8-P8-1']
    # feat_cols = ['FP1-F7', 'F7-T7', 'T7-P7', 'T7-FT9', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'FT10-T8']
    
    class2id = {
        'normal': 0,
        'seizure': 1,
        'preictal': 2,
    }
    
    classes_binary_list = ['normal', 'seizure']
    classes_three_list = ['normal', 'seizure', 'preictal']
    
    used_patients = ['chb01-summary.txt', 'chb03-summary.txt', 'chb07-summary.txt', 'chb09-summary.txt', 
                     'chb10-summary.txt', 'chb20-summary.txt', 'chb21-summary.txt', 'chb22-summary.txt']
    
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.is_three:
            self.classes_list = self.classes_three_list
        else:
            self.classes_list = self.classes_binary_list
        self.readData()
        self.update(self.args.tag)  
        self.classes_weights = None
        torch.multiprocessing.set_sharing_strategy('file_system')
    
    
    def preprocessing(self):
        recordings_path = os.path.join(self.args.root_path, 'recordings')
        summaries_path = os.path.join(self.args.root_path, 'summaries')
        cropped_recordings_path = os.path.join(self.args.root_path, 'cropped_recordings')
        if not os.path.isdir(cropped_recordings_path):
            os.mkdir(cropped_recordings_path)
                
        train_annos = []
        valid_annos = []
        edfs = []
        summaries = os.listdir(summaries_path)
        for summary in tqdm(summaries, desc="Summaries is processing", leave=False):
            if summary not in self.used_patients:
                continue
            edfs.extend(parse(os.path.join(summaries_path, summary)))
        for edf in tqdm(edfs, desc="edfs is processing"):
            if len(edf) == 0:
                continue
            # raw = mne.io.read_raw_edf(os.path.join(recordings_path, edf.file_name))
            raw = mne.io.read_raw_edf(os.path.join(recordings_path, edf.file_name), verbose=False)
            labeled_sample = torch.zeros(raw.get_data().shape[-1])
            for event in edf.events:
                st, et = event.start, event.end
                labeled_sample[int(st*self.args.freq): int(et*self.args.freq)] = 1
            for event in edf.events:
                st, et = event.start, event.end
                annos, t, v = [], [], []
                for i in range(max(0, st-self.args.tolerance), et): # tolerance = 90s, 发病前20s前为正常状态，发病前20s内为预发作状态
                    percent = torch.mean((labeled_sample[int(i * self.args.freq): int((i + 4) * self.args.freq)] == 1).float()).item()
                    
                    if i < max(0, st - self.args.ts) and percent == 0:
                        label = 'normal'
                    elif percent < 0.1:
                        label = 'preictal'
                        if not self.args.is_three:
                            continue
                    elif percent >= 0.1:
                        label = 'seizure'
                    else:
                        raise ValueError
                        
                    annos.append({
                        "file": edf.file_name,
                        "label": label,
                        "st": i,
                        "et": i + self.args.length,
                    })
                    # if label == 'preictal':
                    #     for j in range(3):
                    #         annos.append({
                    #             "file": edf.file_name,
                    #             "label": label,
                    #             "st": i,
                    #             "et": i + self.args.length,
                    #         })
                random.shuffle(annos)
                t, v = train_test_split(annos, train_size=self.args.train_size)
                train_annos.extend(t)
                valid_annos.extend(v)
        print('Data has been annotated entirely')
        return train_annos, valid_annos
            
        
    
    def readData(self):
        self.train_dataset, self.valid_dataset  =  self.preprocessing()
        self.splitedStatistic = {}
        for class_label in self.classes_list:
            self.splitedStatistic.update({
                class_label: {
                    'train': 0,
                    'valid': 0
                }
            })
        for sample in self.train_dataset:
            self.splitedStatistic[sample['label']]['train'] += 1
        for sample in self.valid_dataset:
            self.splitedStatistic[sample['label']]['valid'] += 1
        
        
    def update(self, tag):
        assert tag in ['train', 'valid']
        self.tag = tag
        self.setDataset()
        
    def setDataset(self):
        if self.tag == 'train':
            self.dataset = self.train_dataset
        elif self.tag == 'valid':
            self.dataset = self.valid_dataset
        else:
            raise ValueError
        
    def __len__(self):
        return len(self.dataset)
    
    
    def splitReport(self):
        tb = pt.PrettyTable()
        train_num = 0
        valid_num = 0
        tb.field_names = ['idx', 'Class',  'Training num', 'Validing num']
        try:
            for i, (class_name, static) in enumerate(self.splitedStatistic.items()):
                train_num += static['train']
                valid_num += static['valid']
                tb.add_row([i, class_name, static['train'], static['valid']])
            tb.add_row(['Summary', len(self.splitedStatistic), train_num, valid_num])
        except:
            pass
        return str(tb)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        file_name = data['file']
        label = self.class2id[data['label']]
        st = data['st']
        et = data['et']
        raw = mne.io.read_raw_edf(os.path.join(self.args.root_path, 'recordings', file_name), verbose=False)
        cutted_edf = raw.crop(st, et, include_tmax=False, verbose=False)
        sample_df = cutted_edf.load_data(verbose=False).to_data_frame()
        sample = sample_df[self.feat_cols].values.transpose()
        
        return torch.tensor(sample, dtype=torch.float32), torch.tensor([label]).long()
        
    def collateFn4Cla(self, data):
        samples, labels = zip(*data)
        samples = torch.stack(samples, 0)
        labels = torch.concat(labels, 0)
        return samples, labels.long()