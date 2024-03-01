import copy
import glob
import os
import pickle
import random
import re
import shutil
import librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils import data
import torchaudio

from DatasetManager.wavetables.wavetable_helper import WavetableIteratorGenerator


class WavetableDataset(data.Dataset):
    """
    Wavetable Dataset (Serum)
    Wavetables are sequences of single-cycle waveforms of length 2058 samples.
    """
    def __init__(self, iterator_gen, num_frames, transformations):
        """
        All transformations
        {
            'time_shift': True,
            'time_dilation': True,
            'transposition': True
        }

        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        """
        super().__init__()
        self.split = None
        self.list_ids = {'train': [], 'validation': [], 'test': []}
        self.iterator_gen = iterator_gen
        self.num_frames = num_frames
        self.samples_per_frame = 2048
        self.transformations = transformations

        #  Building/loading the dataset
        if os.path.isfile(self.dataset_file):
            self.load()
        else:
            print(f'Building dataset {str(self)}')
            self.make_tensor_dataset()

    def __str__(self):
        prefix = str(self.iterator_gen)
        name = f'Wavetable-' \
               f'{prefix}-' \
               f'{self.num_frames}'
        return name

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_ids[self.split])

    @property
    def data_folder_name(self):
        # Same as __str__ but without the sequence_len
        name = str(self)
        return name

    @property
    def cache_dir(self):
        cache_dir = f'{os.path.expanduser("~")}/Data/dataset_cache/Wavetable'
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        return cache_dir

    @property
    def dataset_file(self):
        dataset_dir = f'{self.cache_dir}/{str(self)}.txt'
        return dataset_dir

    def save(self):
        # Only save list_ids
        with open(self.dataset_file, 'wb') as ff:
            pickle.dump(self.list_ids, ff, 2)

    def load(self):
        """
        Load a dataset while avoiding local parameters specific to the machine used
        :return:
        """
        with open(self.dataset_file, 'rb') as ff:
            list_ids = pickle.load(ff)
        self.list_ids = list_ids

    def __getitem__(self, index):
        """
        Generates one sample of data
        """
        id = self.list_ids[self.split][index]
        wt_mmap = np.memmap(
            f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/wt',
            dtype=np.float32,
            mode='r',
            shape=(self.num_frames, self.samples_per_frame))
        wt = wt_mmap
        del wt_mmap

        # data augmentations (only for train split) ?? What would it be ?
        # if self.transformations and (self.split == 'train'):

        # Add pad, start and end wavetables ??

        # Tokenize

        return {
            'wt': torch.tensor(wt).long(),
        }

    def iterator_gen(self):
        return (elem for elem in self.iterator_gen())

    def split_datasets(self, split=None, indexed_datasets=None):
        train_dataset = copy.copy(self)
        train_dataset.split = 'train'
        val_dataset = copy.copy(self)
        val_dataset.split = 'validation'
        test_dataset = copy.copy(self)
        test_dataset.split = 'test'
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

    def data_loaders(self,
                     batch_size,
                     num_workers,
                     shuffle_train=True,
                     shuffle_val=False):
        """
        Returns three data loaders obtained by splitting
        self.tensor_dataset according to split
        :param num_workers:
        :param shuffle_val:
        :param shuffle_train:
        :param batch_size:
        :param split:
        :return:
        """

        datasets = self.split_datasets()

        train_dl = data.DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_dl = data.DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=shuffle_val,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        test_dl = data.DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return {'train': train_dl, 'val': val_dl, 'test': test_dl}

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print('Making tensor dataset')

        chunk_counter = {
            'train': 0,
            'validation': 0,
            'test': 0,
        }

        # Build x folder if not existing
        if not os.path.isfile(
                f'{self.cache_dir}/{self.data_folder_name}/xbuilt'):
            if os.path.isdir(f'{self.cache_dir}/{self.data_folder_name}'):
                shutil.rmtree(f'{self.cache_dir}/{self.data_folder_name}')
            os.mkdir(f'{self.cache_dir}/{self.data_folder_name}')
            os.mkdir(f'{self.cache_dir}/{self.data_folder_name}/train')
            os.mkdir(f'{self.cache_dir}/{self.data_folder_name}/validation')
            os.mkdir(f'{self.cache_dir}/{self.data_folder_name}/test')
            # Iterate over files
            for wt_file, split in tqdm(self.iterator_gen()):
                # wav to wavetable
                wt = self.process_wave(wt_file)
                if wt is None:
                    continue
                wt_name = f"{re.split('/', wt_file)[-2]}_{os.path.splitext(re.split('/', wt_file)[-1])[0]}"
                folder_name = f'{self.cache_dir}/{self.data_folder_name}/{split}/{wt_name}'
                if os.path.exists(folder_name):
                    print(f'Skipped {folder_name}')
                    continue
                os.mkdir(folder_name)
                wt_mmap = np.memmap(f'{folder_name}/wt',
                                    dtype=np.float32,
                                    mode='w+',
                                    shape=(self.num_frames,
                                           self.samples_per_frame))
                wt_mmap[:] = np.asarray(wt[:]).astype(np.float32)
                del wt_mmap
            open(f'{self.cache_dir}/{self.data_folder_name}/xbuilt',
                 'w').close()

        # Build index of files
        for split in ['train', 'validation', 'test']:
            paths = glob.glob(
                f'{self.cache_dir}/{self.data_folder_name}/{split}/*')
            for path in paths:
                # read file
                # with open(f'{path}/length.txt', 'r') as ff:
                #     sequence_length = int(ff.read())
                score_name = path.split('/')[-1]

                chunk_counter[split] += 1
                self.list_ids[split].append({
                    'score_name': score_name,
                })

        print(f'Wavetables: {chunk_counter}\n')

        # Save class (actually only serve for self.list_ids, helps with reproducibility)
        self.save()
        return

    def process_wave(self, wt_file):
        wt = torch.tensor(librosa.load(path=wt_file, sr=44100)[0])
        if wt.size(0) % self.samples_per_frame != 0:
            print(
                'Skipping, wrong number of frames or wrong number of samples per frame'
            )
            return None
        wt_split = wt.view(-1, self.samples_per_frame)
        # downsample wavetable res if needed
        if wt_split.shape[0] > self.num_frames:
            indices_downsampled = np.arange(0,
                                            wt_split.shape[0],
                                            step=(wt_split.shape[0] //
                                                  self.num_frames))
            indices_downsampled = indices_downsampled[:32]
            wt_split = wt_split[indices_downsampled]
        elif wt_split.shape[0] < self.num_frames:
            print(f'Skipping, could only find {wt_split.size(0)} frames')
            return None
        # self.visualize_wt(wt_split,
        #                   name=os.path.splitext(os.path.basename(wt_file))[0])
        assert wt_split.shape == (self.num_frames, self.samples_per_frame)
        if torch.abs(wt_split).sum() == 0:
            print(f'Skipping, empty file?')
            return None
        return wt_split

    def tensor_to_wavetable(self, sequences, fill_features):
        """
        Input: torch tensor
        Output: wavetable as a wave file
        """
        raise NotImplementedError

    def visualize_wt(self, wt, name):
        fig = plt.figure(figsize=(8, 8), facecolor='black')
        ax = plt.subplot(frameon=False)
        X = np.linspace(-1, 1, wt.shape[-1])
        lines = list(wt)
        for i in range(len(wt)):
            xscale = 1
            line, = ax.plot(xscale * X, 2 * i + wt[i].numpy(), color="w")
            lines.append(line)
        # Set y limit (or first line is cropped because of thickness)
        # ax.set_ylim(-1, 70)
        # No ticks
        ax.set_xticks([])
        ax.set_yticks([])
        # # 2 part titles to get different font weights
        # ax.text(0.5, 1.0, "Wavetable", transform=ax.transAxes,
        #         ha="right", va="bottom", color="w",
        #         family="sans-serif", fontweight="light", fontsize=16)
        # ax.text(0.5, 1.0, name, transform=ax.transAxes,
        #         ha="left", va="bottom", color="w",
        #         family="sans-serif", fontweight="bold", fontsize=16)
        plt.savefig('dump/wavetable.pdf')

    def visualise_batch(self, piano_sequences, writing_dir, filepath):
        raise NotImplementedError
        # # data is a matrix (batch, ...)
        # # Visualise a few examples
        # if len(piano_sequences.size()) == 1:
        #     piano_sequences = torch.unsqueeze(piano_sequences, dim=0)

        # num_batches = len(piano_sequences)

        # for batch_ind in range(num_batches):
        #     midipath = f"{writing_dir}/{filepath}_{batch_ind}.mid"
        #     score = self.tensor_to_score(sequence=piano_sequences[batch_ind],
        #                                  selected_features=None)
        #     score.write(midipath)


if __name__ == '__main__':
    num_elements = None
    iterator_gen = WavetableIteratorGenerator(num_elements=num_elements)
    num_frames = 32
    transformations = False
    dataset = WavetableDataset(iterator_gen=iterator_gen,
                               num_frames=num_frames,
                               transformations=transformations)

    dataloaders = dataset.data_loaders(batch_size=32,
                                       num_workers=0,
                                       shuffle_train=True,
                                       shuffle_val=True)

    for x in dataloaders['train']:
        print('yoyo')
