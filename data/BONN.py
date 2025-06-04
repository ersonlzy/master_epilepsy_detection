import os
import torch
import numpy as np
from argparse import Namespace
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import prettytable as pt





class Bonn(Dataset):
    label_list = ['A_Z', 'B_O', 'C_N', 'D_F', 'E_S']
    
    label2class_abcd_e = {
        'A_Z': "normal",
        'B_O': "normal",
        'C_N': "normal",
        'D_f': "normal",
        'E_S': "ictal",
    }

    label2class_cd_e = {
        'C_N': "interictal",
        'D_f': "interictal",
        'E_S': "ictal",
    }

    label2class_ab_cd = {
        'A_Z': "normal",
        'B_O': "normal",
        'C_N': "epileptic",
        'D_f': "epileptic",
    }

    label2class_ab_cd_e = {
        'A_Z': "normal",
        'B_O': "normal",
        'C_N': "interictal",
        'D_f': "interictal",
        'E_S': "ictal",
    }


    class2id_abcd_e = {
        'normal': 0,
        'normal': 0,
        'normal': 0,
        'normal': 0,
        'ictal': 1,
    }

    class2id_cd_e = {
        'interictal': 0,
        'interictal': 0,
        'ictal': 1,
    }

    class2id_ab_cd = {
        'normal': 0,
        'normal': 0,
        'epileptic': 1,
        'epileptic': 1,
    }

    class2id_ab_cd_e = {
        'normal': 0,
        'normal': 0,
        'interictal': 1,
        'interictal': 1,
        'ictal': 2,
    }

    class_list = [
        ["normal", "ictal"],
        ["interictal", 'ictal'],
        ['normal', "epileptic"],
        ["normal", "interictal", "ictal"],
    ]
    class2id_list = [class2id_abcd_e, class2id_cd_e, class2id_ab_cd, class2id_ab_cd_e]
    label2class_list = [label2class_abcd_e, label2class_cd_e, label2class_ab_cd, label2class_ab_cd_e]
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.class2id = self.class2id_list[self.args.datasets_task]
        self.classes_list = self.class_list[self.args.datasets_task]
        self.label2class = self.label2class_list[self.args.datasets_task]
        self.readData()
        self.tag = self.args.tag
        self.setDataset()
        
        
        
    def readData(self):
        self.annos = []
        sets = os.listdir(self.args.root_path)
        annos_dict = {c: [] for c in self.classes_list}
        for label in sets:
            if label not in list(self.label2class.keys()):
                continue
            files = os.listdir(os.path.join(self.args.root_path, label))
            for file in files:
                if "._" in file:
                    continue
                for idx in range(0, 4097 - 512, 64):
                    annos_dict[self.label2class[label]].append({
                        'file_name': os.path.join(label, file),
                        'label': self.label2class[label],
                        'st': idx,
                        'et': idx + 512,
                    })
                    # print(annos_dict)
                    # self.annos.append({
                    #     'file_name': os.path.join(label, file),
                    #     'label': label,
                    #     'st': idx,
                    #     'et': idx + 512,
                    # })
        self.annos = self.balance(annos_dict)
        self.train_dataset, self.valid_dataset = train_test_split(self.annos, train_size=self.args.train_size)
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

    def balance(self, annos_dict:dict):
        max_num = max([len(v) for v in annos_dict.values()])
        res = []
        for k, v in annos_dict.items():
            res.extend(annos_dict[k] * (max_num // len(annos_dict[k])))
        return res


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
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        file_name = data['file_name']
        label = data['label']
        st = data['st']
        et = data['et']
        file_path = os.path.join(self.args.root_path, file_name)
        data = np.loadtxt(file_path).reshape(1, -1)
        data = torch.tensor(data[:, st:et], dtype=torch.float32)
        c = self.class2id[label]
        return data, torch.tensor([c]).long()

    def collateFn4Cla(self, data):
        samples, labels = zip(*data)
        samples = torch.stack(samples, 0)
        labels = torch.concat(labels, 0)
        return samples, labels