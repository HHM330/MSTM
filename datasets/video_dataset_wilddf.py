import os
import numpy as np
import cv2
import json
import torch
import torch.utils.data as data
import pickle
import pandas as pd
import random
from collections import OrderedDict
from torchvision.datasets import ImageFolder
class WildDF_Dataset(data.Dataset):
    def __init__(self,
                 root,
                 split='train',
                 num_segments=16,
                 transform=None,
                 sparse_span=150,
                 dense_sample=0,
                 test_margin=1.3,
                 frame_selection='faceqnet_best'):   # faceqnet_best   random  middle
        super().__init__()

        self.root = root
        self.split = split
        self.num_segments = num_segments
        self.transform = transform
        self.sparse_span = sparse_span
        self.dense_sample = dense_sample
        self.test_margin = test_margin
        self.frame_selection = frame_selection
        self.parse_dataset_info()

    def parse_dataset_info(self):
        self.real_video_dir = os.path.join(self.root, 'real_train' if self.split == 'train' else 'real_test').replace('\\','/')
        self.fake_video_dir = os.path.join(self.root, 'fake_train' if self.split == 'train' else 'fake_test').replace('\\','/')

        self.real_names = [f for f in os.listdir(self.real_video_dir)]
        self.fake_names = [f for f in os.listdir(self.fake_video_dir)]
        print(f'{self.split} has {len(self.real_names)} real videos and {len(self.fake_names)} fake videos')

        self.dataset_info = [[x, 'real'] for x in self.real_names] + [[x, 'fake'] for x in self.fake_names]
        self.df = pd.read_csv("datasets/best_scores_wilddf.txt", sep=";", header=0, names=["img", "score"])

    def sample_indices_train(self,video_len):
        base_idxs = np.array(range(video_len), dtype=int)
        if self.sparse_span:
            base_idxs = np.linspace(0, video_len - 1, self.sparse_span, dtype=int)
        base_idxs_len = len(base_idxs)
        def over_sample_strategy(total_len):
            if total_len >= self.num_segments:
                offsets = np.sort(random.sample(range(total_len), self.num_segments))
            else:
                inv_ratio = self.num_segments // total_len
                offsets = []
                for idx in range(total_len):
                    offsets.extend([idx] * inv_ratio)
                tail = [total_len - 1] * (self.num_segments - len(offsets))
                offsets.extend(tail)
            return np.asarray(offsets)

        def dense_sample_strategy(total_len):
            if total_len > self.dense_sample:
                start_idx = np.random.randint(0, total_len - self.dense_sample)
                average_duration = self.dense_sample // self.num_segments
                assert average_duration > 1
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                          np.random.randint(average_duration, size=self.num_segments)
                offsets += start_idx
            else:
                offsets = over_sample_strategy(total_len)
            return offsets

        def non_dense_sample_strategy(total_len):
            average_duration = total_len // self.num_segments
            if average_duration > 1:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                          np.random.randint(average_duration, size=self.num_segments)
            else:
                offsets = over_sample_strategy(total_len)
            return offsets

        if self.dense_sample:
            sampled_idxs = dense_sample_strategy(base_idxs_len)
        else:
            sampled_idxs = non_dense_sample_strategy(base_idxs_len)
        return base_idxs[sampled_idxs].tolist()

    def sample_indices_test(self, video_len):
        base_idxs = np.array(range(video_len), dtype=int)
        if self.sparse_span:
            base_idxs = np.linspace(0, video_len - 1, self.sparse_span, dtype=int)
        base_idxs_len = len(base_idxs)

        if self.dense_sample:
            start_idx = max(base_idxs_len // 2 - self.dense_sample // 2, 0)
            end_idx = min(base_idxs_len // 2 + self.dense_sample // 2, base_idxs_len)
            base_idxs = base_idxs[start_idx: end_idx]
            base_idxs_len = len(base_idxs)

        tick = base_idxs_len / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        offsets = base_idxs[offsets].tolist()

        return offsets

    def get_placeholder_sample(self):
        T, C, H, W = 8, 3, 224, 224
        return torch.zeros((T * C, H, W))

    def __getitem__(self, index):
        video_name, video_label = self.dataset_info[index]
        video_path = os.path.join(eval(f'self.{video_label}_video_dir'), video_name)

        vr = []
        for v in sorted(os.listdir(video_path)):
            vr.append(cv2.imread(os.path.join(video_path, v)))
        video_len = len(vr)

        if self.split == 'train':
            sampled_idxs = self.sample_indices_train(video_len)
        else:
            sampled_idxs = self.sample_indices_test(video_len)

        try:
            frames = [vr[i] for i in sampled_idxs]
        except Exception as e:
            print(f"Error decoding frames for video {video_path}: {e}")
            return self.get_placeholder_sample()

        if self.transform is not None:
            if random.random() < 0.5:
                frames = frames[::-1]

            tmp_imgs = {"image": frames[0]}
            additional_targets = {}
            for i in range(1, len(frames)):
                additional_targets[f"image{i}"] = "image"
                tmp_imgs[f"image{i}"] = frames[i]

            self.transform.add_targets(additional_targets)

            try:
                frames = self.transform(**tmp_imgs)
                frames = OrderedDict(sorted(frames.items(), key=lambda x: x[0]))
                frames = list(frames.values())
                frames = torch.stack(frames)
                process_imgs = frames.view(-1, frames.size(2), frames.size(3)).contiguous()
            except Exception as e:
                print(f"Error transforming frames for video {video_path}: {e}")
                return self.get_placeholder_sample()
        else:
            process_imgs = frames

        video_label_int = 0 if video_label == 'real' else 1

        video_path_norm = video_path.replace('\\', '/') + '/'
        pic_df = self.df[self.df['img'].str.contains(video_path_norm)].reset_index(drop=True)

        if self.frame_selection == 'faceqnet_best':
            pic_path = pic_df.iloc[0]['img']
        elif self.frame_selection == 'random':
            pic_path = pic_df.sample(n=1).iloc[0]['img']
        elif self.frame_selection == 'middle':
            mid_idx = len(pic_df) // 2
            pic_path = pic_df.iloc[mid_idx]['img']
        else:
            raise ValueError(f"Unknown frame_selection: {self.frame_selection}")

        max_idxspic = cv2.imread(pic_path)
        max_idxspic = self.transform(image=max_idxspic)['image']

        return process_imgs, video_label_int, video_path, sampled_idxs, max_idxspic

    def __len__(self):
        return len(self.dataset_info)
















