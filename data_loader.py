import multiprocessing as mp
import pandas as pd
import os
import torch
import numpy as np
import cv2
import yaml
import time
import sys
from multiprocessing import Queue
from tqdm import tqdm
from collections import deque
from utils.augmentations import letterbox

class Loader:
    def __init__(self, batch_size, data_yaml, img_size=640, vid_path='nfl-health-and-safety-helmet-assignment/train', label_path='nfl-health-and-safety-helmet-assignment/train_labels.csv'):
        self.vf = os.listdir(vid_path)
        self.vp = vid_path
        self.dataset = Dataset(label_path, data_yaml, img_size)
        self.bs = batch_size
        self.vid_id = 0
        self.frames = deque()
        self.len = len(pd.read_csv(label_path)['video_frame'].unique())
        self.waiting = Queue()
        self.work = Queue()
        self.processes = list()
        self.working = list()
        self.nworkers = os.cpu_count()

    def __len__(self):
        return -(-self.len // self.bs) #ceil divide
    
    def __iter__(self):
        while len(mp.active_children()) < self.nworkers:
            v = mp.Value('i', False)
            p = mp.Process(target=self.dataset.run, args=[self.waiting, self.work, v], daemon=True)
            p.start()
            self.processes.append(p)
            self.working.append(v)
        self.process_vid()
        return self

    def __next__(self):
        self.process_vid()
        while self.vid_id < len(self.vf) and len(self.frames) < self.bs or len(self.frames) < self.bs and self.active_workers() > 0:
            self.process_vid()
            time.sleep(1) #wait for the last of the frames to be loaded
        
        if len(self.frames) == 0 and self.active_workers() == 0:
            raise StopIteration

        frames = None
        if len(self.frames) < self.bs: #uneven number of images left
            frames = [self.frames.pop() for _ in range(len(self.frames))]
        else: 
            frames = [self.frames.pop() for _ in range(self.bs)] #returns list of frame and labels tuple
        frames, labels = zip(*frames) #returns list of frames and list of labels (basically list transpose)
        frames = torch.cat(frames, dim=0) #put frames together into 1 tensor
        for i in range(len(labels)):
            l = labels[i]
            fnum = torch.zeros((len(l),)) + i
            l[:,0] = fnum #change the frame number
        labels = torch.cat(labels, dim=0) #turn into 1 tensor
        return frames, labels
    
    def process_vid(self):
        if len(self.frames) < 10 * self.bs and self.vid_id < len(self.vf) and self.active_workers() < self.nworkers:
            vid = self.vf[self.vid_id]
            self.work.put((self.vp, vid))
            self.vid_id += 1

        while not self.waiting.empty():
            self.frames.append(self.waiting.get())
    
    def active_workers(self):
        return sum(map(lambda x: x.value, self.working))
            
class Dataset:
    def __init__(self, path, data_yaml, img_size) -> None:
        self.img_size = img_size
        labels = pd.read_csv(path)
        labels = labels[labels['isSidelinePlayer'] == False]
        self.len = len(labels['video_frame'].unique())
        labels = labels.drop(['isSidelinePlayer',
                              'isDefinitiveImpact',
                              'impactType',
                              'playID',
                              'gameKey',
                              'video_frame'], axis=1)
        labels['x'] = labels['left'] + labels['width']/2
        labels['y'] = labels['top'] + labels['height']/2
        labels['w'] = labels['width']
        labels['h'] = labels['height']

        #divide xywh by resolution to get range [0-1]
        #reshape images correctly

        labels = labels.drop(['left', 'width', 'top', 'height'], axis=1)
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            data = data['names2']
        convert = dict()
        for i, val in enumerate(data):
            convert[val] = i
        labels['label'] = labels['label'].apply(lambda x: convert[x]) #convert label string to encoding
        self.labels = labels
    
    def run(self, waiting, work, working):
        while True:
            if not work.empty():
                working.value = True
                vp, vid = work.get()
                cap = cv2.VideoCapture(os.path.join(vp, vid))
                mask = self.labels['video'] == vid
                labels = self.labels[mask]
                self.next_vid(cap, waiting, labels)
            working.value = False
            time.sleep(1)
    
    def next_vid(self, cap, waiting, labels):
        fn = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            
            frame, ratio, (dw, dh) = letterbox(frame, new_shape=self.img_size)

            frame = frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            frame = np.ascontiguousarray(frame)
            s = frame.shape #[ch,h,x]
            frame = torch.from_numpy(frame).reshape(1,s[0],s[1],s[2]).pin_memory()
            mask = labels['frame'] == fn
            l = labels[mask]
            l = l.drop(['view', 'video'], axis=1)
            box = l[['x','y','w','h']].astype(np.float32).values
            box = box * ratio[0] #multiply by scale ratio then offset letterbox and set range [0,1]
            box[:,[0,2]] = (box[:,[0,2]] + dw) / s[2]
            box[:,[1,3]] = (box[:,[0,2]] + dh) / s[1]
            l = np.concatenate([l[['frame', 'label']].astype(np.float32).values, box], axis=1)
            l = torch.from_numpy(l).pin_memory()
            fn += 1
            waiting.put((frame, l))

if __name__ == '__main__':
    #tests
    train_labels_path = 'nfl-health-and-safety-helmet-assignment/train_labels.csv'
    temp_yaml = 'temp_yaml.yml'
    print('TEST 1...', end='')
    labels = pd.read_csv('nfl-health-and-safety-helmet-assignment/train_labels.csv')
    labels = labels = labels.drop([
                            'isDefinitiveImpact',
                            'impactType',
                            'playID',
                            'gameKey',
                            'video_frame'], axis=1)
    cls = labels['label'].unique()
    cls = list(cls)
    reverse_cls = dict()
    for i, c in enumerate(cls):
        reverse_cls[i] = c
    d = dict()
    d['names'] = cls
    d['random'] = '91273'
    with open(temp_yaml, 'w') as txt:
        yaml.safe_dump(d,txt,default_flow_style=False)
    d = Dataset(train_labels_path, temp_yaml)
    assert (d.labels['label'].apply(lambda x: reverse_cls[x]) == labels[labels['isSidelinePlayer'] == False]['label']).all()
    print('\t\t\tGOOD')

    print('TEST 2...', end='')
    d = Dataset(train_labels_path, temp_yaml)
    path = 'nfl-health-and-safety-helmet-assignment/train'
    vid = '58104_000352_Sideline.mp4'
    nextvid = d.next_vid(path, vid)
    cap = cv2.VideoCapture(os.path.join(path, vid))
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame)
        frame = frame.permute((2,0,1))
        s = frame.shape
        frame = frame.reshape(1,s[0],s[1],s[2])
        assert (frame == nextvid[i][0]).all()
    print('DONE')
    
    l = Loader(16, temp_yaml)
    print('TEST 3...', end='')
    print('\ttotal batches:', len(l), end='\t')
    print('DONE')
    
    #print('TEST 4', end='')
    #load = iter(l)
    #print(next(load)[1][:,0].int())
    #print('DONE')

    print('TEST5')
    tq = tqdm(l)
    for i in tq:
        assert i != None
    print('DONE')
    os.remove(temp_yaml)
    