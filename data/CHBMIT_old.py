import mne
import warnings
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import yaml
import sklearn.preprocessing as preprocessing
from tqdm import tqdm

warnings.filterwarnings('ignore')


# chbmit_columns = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'VNS', 'F7-CS2', 'T7-CS2', 'P7-CS2', 'FP1-CS2', 'F3-CS2', 'C3-CS2', 'P3-CS2', 'O1-CS2', 'FZ-CS2', 'CZ-CS2', 'PZ-CS2', 'FP2-CS2', 'F4-CS2', 'C4-CS2', 'P4-CS2', 'O2-CS2', 'F8-CS2', 'T8-CS2', 'P8-CS2', 'C2-CS2', 'C6-CS2', 'CP2-CS2', 'CP4-CS2', 'CP6-CS2', 'F7', 'T7', 'P7', 'FP1', 'F3', 'C3', 'P3', '01', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8', 'EKG1-CHIN', 'C2', 'C6', 'CP2', 'CP4', 'CP6', 'LOC-ROC', 'T8-P8', 'PZ-OZ', 'FC1-Ref', 'FC2-Ref', 'FC5-Ref', 'FC6-Ref', 'CP1-Ref', 'CP2-Ref', 'CP5-Ref', 'CP6-Ref', 'ECG']
chbmit_columns = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']
shape_list = [
    [16, 512], #0
    [32, 512], #1
    [64, 512], #2
    [16, 1024], #3
    [32, 1024], #4
    [64, 1024], #5
    [16, 2048], #6
    [32, 2048], #7
    [4 * 256], #8
]

binary_classes_map = {
    'normal': 0,
    'seizure': 1,
}

ternary_classes_map = {
    'normal': 0,
    'seizure': 1,
    'preictal': 2
}

quaternary_classes_map = {
    'normal': 0,
    'seizure': 1,
    'preictal': 2,
    'interictal': 3,
}

class Event():
    def __init__(self, start_time, end_time):
        self.start = start_time
        self.end = end_time


class EDF():
    def __init__(self, file_name, events):
        self.file_name = file_name
        self.events = events
    
    def __len__(self):
        return len(self.events)
        

class DataFunction():
    def __init__(self):
        pass
    
    @staticmethod
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
                    seizure_list.append(Event(st, et - st))
                except IndexError:
                    pass
        return EDF(file_name, seizure_list)

    @staticmethod
    def parse(summary_path):
        edfs = []
        with open(summary_path) as f:
            content = f.read()
            info_list = content.split('\n\n')
            for info in info_list:
                if 'File Name' in info:
                    edfs.append(DataFunction.infoParse(info))
        return edfs
    
    @staticmethod
    def in01(x, y):
        if x < 0:
            x = 0
        if y > 1:
            y = 1
        return x, y
    

    @staticmethod
    def labelDefine(stl, etl, tmin, length):
        nstl, netl, nlabel = [], [], []
        label = 1
        for index in range(len(stl)):
            st = stl[index]
            et = etl[index]
            
            nst = (st - tmin) / length
            net = (et - tmin) / length
            if net < 0 or nst > 1:
                pass
            else:
                # nst, net = DataFunction.in01(nst, net)
                nstl.append(nst)
                netl.append(net)
                nlabel.append(label)
        return nstl, netl, nlabel

    @staticmethod
    def abormalTargetDefine(stl, etl, tmin, length):
        istl, ietl = [], []
        for st, et in zip(stl, etl):
            ist = st - tmin
            iet = et - tmin
            if ist >= length or iet < 0:
                continue
            elif ist < 0:
                ist = 0
                if iet >= length:
                    iet = length
            elif iet >= length:
                iet = length
            else:
                pass
            istl.append(int(ist * 256))
            ietl.append(int(iet * 256) - 1)
        return istl, ietl
                
    # @staticmethod
    # def dataCutOff(edfs, shape_code, root_path='./'):
    #     shape = shape_list[shape_code]
    #     length = shape[0] / 256
    #     annotations = []

    #     recordings_path = os.path.join(root_path, 'CHBMIT', 'recordings')
    #     cutted_recordings_path = os.path.join(root_path, 'CHBMIT', 'cutted_recordings')
    #     if not os.path.isdir(cutted_recordings_path):
    #         os.mkdir(cutted_recordings_path)
    #     annotations_path = os.path.join(root_path, 'CHBMIT', 'annotations', 'annotations.yaml')

    #     for edf in tqdm(edfs, desc="EDF files is processing"):
    #         file_name = edf.file_name.replace('.edf', '')
    #         path = os.path.join(recordings_path, edf.file_name)
    #         raw = mne.io.read_raw_edf(path, verbose=False)
    #         stl, etl= [], []
    #         for event in edf.events:
    #            stl.append(event.st)
    #            etl.append(event.st + event.duration)
    #         tmin = 0
    #         tmax = tmin + length
    #         identify_code = 0
            
    #         while True:
    #             annotation_dict = {}
    #             try:
    #                 cutted_edf = raw.copy().crop(tmin, tmax, include_tmax=False)
    #                 cutted_recording_name =  '{}_{}.csv'.format(file_name, identify_code)
    #                 cutted_edf_path = os.path.join(cutted_recordings_path, cutted_recording_name)

    #                 istl, ietl = DataFunction.abormalTargetDefine(stl, etl, tmin, length)
    #                 annotation_dict['path'] = cutted_edf_path
    #                 annotation_dict['anno'] = {
    #                     'onset': istl,
    #                     'cease': ietl,
    #                 }

    #                 df = cutted_edf.to_data_frame(verbose=False)
    #                 df.drop('time', axis=1)
    #                 for col in chbmit_columns:
    #                     if col not in df.columns.values.tolist():
    #                         df.insert(df.shape[1], col, 0)
    #                 df = df[chbmit_columns]

    #                 # data = df.to_numpy()
    #                 # np.save(cutted_edf_path, data)
    #                 df.to_csv(cutted_edf_path, index=False)
    #                 annotations.append(annotation_dict)

    #                 identify_code += 1
    #                 tmin += 8
    #                 tmax += 8
    #             except ValueError:
    #                 break
            
    #     with open(annotations_path, 'w+', encoding='utf-8') as f:
    #         yaml.dump(data=annotations, stream=f, allow_unicode=True)
    #     print('Data has been cutted off entirely')
            

    @staticmethod
    def dataCutOff(edfs, shape_code, root_path='./'):
        shape = shape_list[shape_code]
        length = shape[0] / 256
        annotations = []

        recordings_path = os.path.join(root_path, 'CHBMIT', 'recordings')
        cutted_recordings_path = os.path.join(root_path, 'CHBMIT', 'cutted_recordings')
        if not os.path.isdir(cutted_recordings_path):
            os.mkdir(cutted_recordings_path)
        annotations_path = os.path.join(root_path, 'CHBMIT', 'annotations', 'annotations.yaml')

        for edf in tqdm(edfs, desc="EDF files is processing"):
            file_name = edf.file_name.replace('.edf', '')
            path = os.path.join(recordings_path, edf.file_name)
            raw = mne.io.read_raw_edf(path, verbose=False)
            stl, etl= [], []
            for event in edf.events:
               stl.append(event.st)
               etl.append(event.st + event.duration)
            if edf.events:
                event_code = 0
                for event in edf.events:
                    interval_time = 5 * length
                    onset = event.st
                    cease = event.st + event.duration

                    tmin = onset - interval_time
                    if tmin < 0:
                        tmin = 0
                    tmax = tmin + length

                    identify_code = 0
                    while tmax <= cease + interval_time:
                        annotation_dict = {}
                        try:
                            cutted_edf = raw.copy().crop(tmin, tmax, include_tmax=False)

                            cutted_recording_name =  '{}_{}_{}.csv'.format(file_name, event_code, identify_code)
                            cutted_edf_path = os.path.join(cutted_recordings_path, cutted_recording_name)

                            istl, ietl = DataFunction.abormalTargetDefine(stl, etl, tmin, length)
                            annotation_dict['path'] = cutted_edf_path
                            annotation_dict['anno'] = {
                                'onset': istl,
                                'cease': ietl,
                            }

                            df = cutted_edf.to_data_frame(verbose=False)
                            df.drop('time', axis=1)
                            for col in chbmit_columns:
                                if col not in df.columns.values.tolist():
                                    df.insert(df.shape[1], col, 0)
                            df = df[chbmit_columns]

                            # data = df.to_numpy()
                            # np.save(cutted_edf_path, data)
                            df.to_csv(cutted_edf_path, index=False)
                            annotations.append(annotation_dict)

                            identify_code += 1
                            tmin += 2
                            tmax += 2
                        except ValueError:
                            break
            else:
                    pass
            
        with open(annotations_path, 'w+', encoding='utf-8') as f:
            yaml.dump(data=annotations, stream=f, allow_unicode=True)
        print('Data has been cutted off entirely')
        
    @staticmethod
    def cutOff_CxHxW(edfs, shape_code, root_path='./'):
        shape = shape_list[shape_code]
        length = (shape[0] * shape[1]) / 256
        annotations = []

        recordings_path = os.path.join(root_path, 'CHBMIT', 'recordings')
        cutted_recordings_path = os.path.join(root_path, 'CHBMIT', 'cutted_recordings')
        annotations_path = os.path.join(root_path, 'CHBMIT', 'annotations', 'annotations.yaml')

        for edf in edfs:
            file_name = edf.file_name.replace('.edf', '')
            path = os.path.join(recordings_path, edf.file_name)
            raw = mne.io.read_raw_edf(path)
            stl, etl = [], []
            for event in edf.events:
               stl.append(event.st)
               etl.append(event.st + event.duration)
            tmin = 0
            tmax = length
            identify_code = 0
            while True:
                annotation_dict = {}
                # nstl = []
                # netl = []
                # nlabel = []
                
                try:
                    cutted_edf = raw.copy().crop(tmin, tmax, include_tmax=False)
                except ValueError:
                    break
                
                nstl, netl, nlabel = DataFunction.labelDefine(stl, etl, tmin, length)

                cutted_recording_name =  '{}_{}.npy'.format(file_name, identify_code)
                cutted_edf_path = os.path.join(cutted_recordings_path, cutted_recording_name)

                data = cutted_edf.get_data()

                annotation_dict['path'] = cutted_edf_path
                annotation_dict['anno'] = {
                    'onset': nstl,
                    'cease': netl,
                    'label': nlabel
                }

                np.save(cutted_edf_path, data)
                identify_code += 1
                annotations.append(annotation_dict)

                tmin = tmax
                tmax += length
        
        with open(annotations_path, 'w+', encoding='utf-8') as f:
            yaml.dump(data=annotations, stream=f, allow_unicode=True)

    @staticmethod
    def cutOff_CxHxW_ForDetection(edfs, shape_code, root_path='./'):
        shape = shape_list[shape_code]
        length = (shape[0] * shape[1]) / 256
        annotations = []

        recordings_path = os.path.join(root_path, 'CHBMIT', 'recordings')
        cutted_recordings_path = os.path.join(root_path, 'CHBMIT', 'cutted_recordings')
        if not os.path.isdir(cutted_recordings_path):
            os.mkdir(cutted_recordings_path)
        annotations_path = os.path.join(root_path, 'CHBMIT', 'annotations', 'annotations.yaml')

        for edf in tqdm(edfs, desc="EDF files is processing"):
            file_name = edf.file_name.replace('.edf', '')
            path = os.path.join(recordings_path, edf.file_name)
            raw = mne.io.read_raw_edf(path, verbose=False)
            stl, etl= [], []
            for event in edf.events:
               stl.append(event.st)
               etl.append(event.st + event.duration)
            if edf.events:
                event_code = 0
                for event in edf.events:
                    interval_time = length - 5
                    onset = event.st
                    cease = event.st + event.duration

                    tmin = onset - interval_time
                    tmax = tmin + length

                    identify_code = 0
                    while tmax < cease + interval_time:
                        annotation_dict = {}
                        try:
                            cutted_edf = raw.copy().crop(tmin, tmax, include_tmax=False)

                            cutted_recording_name =  '{}_{}_{}.npy'.format(file_name, event_code, identify_code)
                            cutted_edf_path = os.path.join(cutted_recordings_path, cutted_recording_name)

                            nstl, netl, nlabel = DataFunction.labelDefine(stl, etl, tmin, length)
                            annotation_dict['path'] = cutted_edf_path
                            annotation_dict['anno'] = {
                                'onset': nstl,
                                'cease': netl,
                                'label': nlabel
                            }

                            df = cutted_edf.to_data_frame(verbose=False)
                            df.drop('time', axis=1)
                            for col in chbmit_columns:
                                if col not in df.columns.values.tolist():
                                    df.insert(df.shape[1], col, 0)
                            df = df[chbmit_columns]

                            data = df.to_numpy()
                            np.save(cutted_edf_path, data)
                            annotations.append(annotation_dict)

                            identify_code += 1
                            tmin += 10
                            tmax += 10
                        except ValueError:
                            break
            else:
                    pass
            
        with open(annotations_path, 'w+', encoding='utf-8') as f:
            yaml.dump(data=annotations, stream=f, allow_unicode=True)
        print('Data has been cutted off entirely')


    @staticmethod
    def transformToHxW(data, shape_code):
        scaler = preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data.T).T
        data = torch.tensor(data).transpose(-1, -2)
        shape = shape_list[shape_code]

        data = data.view(data.shape[0], *shape)
        return data

    @staticmethod
    def targetTransformForDetectTask(target, data):
        onset = target['onset']
        cease = target['cease']
        label = target['label']
        segments = torch.tensor([onset, cease]).T
        label = torch.tensor(label).T
        target = {
            'segment_set': segments,
            'class_set': label
        }
        return target
    
    @staticmethod
    def transform4AbnormalDetection(anno, data):
        onsets = anno['onset']
        ceases = anno['cease']
        
        target = torch.zeros([data.shape[-1]])
        for onset, cease in zip(onsets, ceases):
            target[onset: cease] = 1
        return target
    
    def transform4Classification(anno, data):
        onsets = anno['onset']
        if onsets:
            target = torch.tensor([1])
        else:
            target = torch.tensor([0])
        return target
            
            

class CHBMIT(Dataset):
    cut_off_method_list = [
        None,
        DataFunction.cutOff_CxHxW,
        DataFunction.cutOff_CxHxW_ForDetection,
        DataFunction.dataCutOff,
    ]

    transform_list = [
        None,
        DataFunction.transformToHxW
    ]

    target_transform_list = [
        None,
        DataFunction.targetTransformForDetectTask,
        DataFunction.transform4AbnormalDetection,
        DataFunction.transform4Classification
    ]


    def __init__(self, shape_code=None, cut_off_method=0, transform=0, target_transform=0, topk=-1, root_path='./'):
        super().__init__()
        self.transform = self.transform_list[transform]
        self.target_transform = self.target_transform_list[target_transform]
        self.shape_code = shape_code
        edfs = []


        annotations_path = os.path.join(root_path, 'CHBMIT', 'annotations')
        annotations_file_path = os.path.join(root_path, 'CHBMIT', 'annotations', 'annotations.yaml')
        summaries_path = os.path.join(root_path, 'CHBMIT', 'summaries')
        if not os.path.isdir(annotations_path):
            os.mkdir(annotations_path)
        annotations_list = os.listdir(annotations_path)
        if not annotations_list:
            summaries_list = os.listdir(summaries_path)
            for summary in tqdm(summaries_list, desc='Summary is processing'):
                summary_path = os.path.join(summaries_path, summary)
                edfs.extend(DataFunction.parse(summary_path))
            if cut_off_method:
                self.cut_off_method_list[cut_off_method](edfs, self.shape_code, root_path)
            os.system('clear')
        with open(annotations_file_path, 'r', encoding='utf-8') as f:
            self.annos = list(yaml.load_all(f.read(), Loader=yaml.FullLoader))[0]

        # try:
        #     eig_vector = np.load('./datasets/CHBMIT/eig_vector.npy')
        # except FileNotFoundError:
        #     eig_vector = self.PCA()
        # self.eig_vector = eig_vector[:,: topk]
        # self.calCalssNum()

    def PCA(self):
        cov_sum = np.zeros((84, 84))
        for i in tqdm(range(len(self)), desc='PCA is processing'):
            anno = self.annos[i]
            data = np.load(anno['path'])
            mean = data.mean(0)
            mean = mean[np.newaxis, :]
            data = data - np.tile(mean, (data.shape[0], 1))
            cov = np.matmul(data.T, data) / (data.shape[-1] - 1)
            cov_sum += cov
        eig_value, eig_vector = np.linalg.eig(cov_sum)

        order = np.argsort(eig_value)
        order = order[::-1]
        eig_vector = eig_vector[:, order]
        np.save("./datasets/CHBMIT/eig_vector.npy", eig_vector)
        scaler = preprocessing.MinMaxScaler()
        eig_vector = scaler.fit_transform(eig_vector.T).T
        return eig_vector

    def calCalssNum(self):
        c0 = 0
        c1 = 0
        criterion = torch.tensor([0])
        for anno in self.annos:
            df = pd.read_csv(anno['path'])
            data = df.to_numpy()
            target = anno['anno']
            target = self.target_transform(target, data)
            if target == criterion:
                c0 += 1
            else:
                c1 += 1
        print('Class 0 num: ', c0, 'Class 1 num: ', c1)

            
    def __getitem__(self, index):
        anno = self.annos[index]
        # data = np.load(anno['path'])
        df = pd.read_csv(anno['path'])
        data = df.to_numpy()
        target = anno['anno']

        # data = np.matmul(data, self.eig_vector)
        if self.transform:
            data = self.transform(data, self.shape_code)
        if self.target_transform:
            target = self.target_transform(target, data)
        return data, target

    def __len__(self):
        return len(self.annos)
    



if __name__ == '__main__':
    F = DataFunction()

    chbmit = CHBMIT(
        shape_code = 8,
        cut_off_method = 3,
        transform = 1,
        target_transform = 3,
        root_path='/mnt/data/intern/lzy/chbmit'
    )
