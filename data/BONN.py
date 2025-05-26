import os
import torch
import numpy as np
from argparse import Namespace
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import prettytable as pt





class Bonn(Dataset):
    label_list = ['A_Z', 'B_O', 'C_N', 'D_F', 'E_S']
    
    # A vs B vs C vs D vs E
    class_list_5 = ['A_Z', 'B_O', 'C_N', 'D_F', 'E_S']
    
    class2id_5 = {
        'A_Z': 0,
        'B_O': 1,
        'C_N': 2,
        'D_F': 3,
        'E_S': 4
    }
    
    label2class_5 = {
        'A_Z': 'A_Z',
        'B_O': 'B_O',
        'C_N': 'C_N',
        'D_F': 'D_F',
        'E_S': 'E_S',
    }
    
    # AB vs CDE
    class_list_2 = ['normal', 'epileptic']
    class2id_2 = {
        'normal': 0,
        'epileptic': 1
    }
    
    label2class_2 = {
        'A_Z': 'normal',
        'B_O': 'normal',
        'C_N': 'epileptic',
        'D_F': 'epileptic',
        'E_S': 'epileptic',
    }
    
    # AB vs CD vs E
    class_list_3 = ['normal', 'ictal', 'interictal']
    
    class2id_3 = {
        'normal': 0,
        'ictal': 1,
        'interictal': 2
    }
    label2class_3 = {
        'A_Z': 'normal',
        'B_O': 'normal',
        'C_N': 'interictal',
        'D_F': 'interictal',
        'E_S': 'ictal',
    }
    
    
    
    # AB vs CD
    class_list_4 = ['normal', 'epileptic']
    
    class2id_4 = {
        'normal': 0,
        'epileptic': 1
    }
    label2class_4 = {
        'A_Z': 'normal',
        'B_O': 'normal',
        'C_N': 'epileptic',
        'D_F': 'epileptic',
    }
    
    # CD vs E
    class_list_1 = ['ictal', 'interictal']
    class2id_1 = {
        'interictal': 0,
        'ictal': 1,
    }
    
    label2class_1 = {
        'C_N': 'interictal',
        'D_F': 'interictal',
        'E_S': 'ictal',
    } 
    
    class_list_dict = {
        1: class_list_1,
        4: class_list_4,
        5: class_list_5,
        2: class_list_2,
        3: class_list_3
    }
    
    class2id_dict = {
        1: class2id_1,
        4: class2id_4,
        5: class2id_5,
        2: class2id_2,
        3: class2id_3
    }
    
    label2class_dict = {
        1: label2class_1,
        4: label2class_4,
        5: label2class_5,
        2: label2class_2,
        3: label2class_3
    }
    
    weights_dict = {
        5: torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32),
        2: torch.tensor([1.3, 1], dtype=torch.float32),
        3: torch.tensor([1, 1, 2], dtype=torch.float32)
    }
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.class2id = self.class2id_dict[self.args.num_classes]
        self.classes_list = self.class_list_dict[self.args.num_classes]
        self.label2class = self.label2class_dict[self.args.num_classes]
        self.readData()
        # self.classes_weights = self.weights_dict[self.args.num_classes]
        self.classes_weights = None
        self.tag = self.args.tag
        self.setDataset()
        
        
        
    def readData(self):
        self.annos = []
        sets = os.listdir(self.args.root_path)
        for label in sets:
            if label not in list(self.label2class.keys()):
                continue
            files = os.listdir(os.path.join(self.args.root_path, label))
            for file in files:
                for idx in range(0, 4097 - 512, 64):
                    self.annos.append({
                        'file_name': os.path.join(label, file),
                        'label': label,
                        'st': idx,
                        'et': idx + 512,
                    })
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
            self.splitedStatistic[self.label2class[sample['label']]]['train'] += 1
        for sample in self.valid_dataset:
            self.splitedStatistic[self.label2class[sample['label']]]['valid'] += 1
            
            
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
        c = self.class2id[self.label2class[label]]
        return data, torch.tensor([c]).long()

    def collateFn4Cla(self, data):
        samples, labels = zip(*data)
        samples = torch.stack(samples, 0)
        labels = torch.stack(labels, 0)
        return samples, labels, None