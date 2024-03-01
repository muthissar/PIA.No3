from os import name
import random
from DatasetManager.piano.piano_helper import find_nearest_value, get_time_table_duration
from collections import defaultdict
import json
import numpy as np
from pathlib import Path
import shutil
import time
from tqdm import tqdm
import warnings

import pretty_midi

pretty_midi.pretty_midi.MAX_TICK = 1e16

import torch
from torch.utils.data import Dataset

START_SYMBOL = 'START'
END_SYMBOL = 'END'
PAD_SYMBOL = 'XX'


class NESDataset(Dataset):
    def __init__(self, phase='train', voices=None):
        self.root = (Path(__file__) / '../../../../data/nesmdb_midi').resolve()
        self.voices = voices if voices is not None else [0, 1, 2, 3]

        self.train = phase == 'train'

        if not self.root.exists():
            raise ValueError(
                f"The dataset could not be found in this folder: {self.root}.\n"
                "Move it to that folder or create a link if you already have it elsewhere.\n"
                "Otherwise run ./DatasetManager/nes/init_nes_dataset.sh to download and extract it (72 MB)."
            )

        self.processed = self.root / 'processed'
        self.processed.mkdir(exist_ok=True)

        # assign an id to each instrument
        self.instrument_to_id = {80: 0, 81: 1, 38: 2, 121: 3}
        self.id_to_instrument = {
            v: k
            for k, v in self.instrument_to_id.items()
        }

        # Loading index dicts
        self.dict_path = self.root / 'pitches.json'
        if self.dict_path.exists():
            with open(self.dict_path, 'r') as f:
                pitches_dict = json.load(f)
                # JSON only supports type string for dict keys
                self.pitches_dict = {
                    int(k): v
                    for k, v in pitches_dict.items()
                }

        else:
            self.pitches_dict = defaultdict(
                lambda: len(self.pitches_dict.keys()), {
                    -1: -1,
                    0: 0
                })  # enable automatic updating of keys

            shutil.rmtree(self.processed)

            self.processed.mkdir(exist_ok=True)

            # preprocessing
            print(f'Preprocessing {self}...')
            t0 = time.time()
            self.preprocess_dataset()
            t1 = time.time()
            d = t1 - t0
            print(
                'Done. Time elapsed:', '{:.0f} s.'.format(d)
                if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))

            self.pitches_dict = dict(
                self.pitches_dict)  # remove defaultdict behaviour
            with open(self.dict_path, 'w') as f:
                json.dump(self.pitches_dict, f)

        self.paths = list((self.processed / phase).glob('*.npy'))

        self.data_augmentation = lambda x: x

    def __getitem__(self, idx):

        path = self.paths[idx]
        # print(path)
        score = np.load(path, allow_pickle=False)[:, self.voices, :]

        ## data augmentation
        # score = self.data_augmentation(score)
        if self.train:
            padding_mask = (score == -1)

            # 1. transpose melodic voices by a random number of semitones between -6 and 5
            actual_voices = [
                i for i in range(score.shape[1]) if score[0, i, 0] >= 0
            ]
            melodic_voices = [
                i for i in actual_voices
                if self.id_to_instrument[self.voices[i]] < 112
            ]  # TODO:

            if melodic_voices != []:
                melodic_pitches = score[:, melodic_voices, 0] % 128
                pitch_shift = np.random.randint(
                    -min(6, melodic_pitches[melodic_pitches > 0].min()),
                    min(6, 128 - melodic_pitches.max()))
                score[:, melodic_voices, 0] += pitch_shift

            # 2. adjust the speed of the piece by a random percentage between +/- 5%
            time_speed = 1 + (np.random.random() - 0.5) / 10

            score[:, :, 1] *= time_speed
            score[:, :, 3] *= time_speed
            #
            # actual_num_voices = sum(data[0,:,0] == 0)
            # if actual_num_voices > 1 and np.random.random() < 0.5:
            #     # 3. half of the time, remove one of the instruments from the ensemble
            #     score[:,np.random.randint(actual_num_voices)] = 0
            #
            #     # 4. half of the time, shuffle the score-to-instrument alignment for the melodic instruments only
            #     melodic_voices = max(actual_num_voices, 3)
            #     # v1 = np.random.randint(melodic_voices)
            #     # v2 = (v1 + 1) % melodic_voices
            #     # # score[:,v1] # TODO:
            # score[padding_mask] = -1

        # replace midi pitches by an id so that every id between 0 and id_max are used
        score[:, :, 0] = np.vectorize(self.pitches_dict.__getitem__)(
            score[:, :, 0].astype(np.int32))
        return torch.from_numpy(score)

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        return f'<NESDataset(dir={self.processed})>'

    def __str__(self):
        return 'NES-MDB Dataset'

    # PREPROCESSING

    def preprocess_dataset(self):
        r"""Recursively explore the dataset and generate for each MIDI file found
        a numpy array containing its data
        """
        bar = tqdm(sorted(list(self.root.glob('**/*.mid'))), leave=False)
        for src_file in bar:
            fname = src_file.name
            bar.set_description(fname[:29] +
                                '...' if len(fname) > 32 else fname +
                                (32 - len(fname)) * ' ')

            target_dir = self.processed / src_file.relative_to(
                self.root).parent
            target_name = src_file.stem + '.npy'
            target_file = target_dir / target_name

            # skip already preprocessed files
            if target_file.exists():
                continue

            target_dir.mkdir(exist_ok=True)

            # compute musical events as tensor
            try:
                midi = pretty_midi.PrettyMIDI(str(src_file))
                tensor = self.parse_midi(midi)
            except ValueError:
                warnings.warn(
                    f"Some error occurred during the preprocessing of {src_file}"
                )

            # save the computed tensor
            np.save(target_file, tensor, allow_pickle=False)

    def parse_midi(self, midi):
        r"""Read a MIDI file and return all notes in it in a numpy.ndarray

        Arguments:
            midi (pretty_midi.PrettyMIDI): MIDI file to be read

        Returns:
            numpy.ndarray: padded content of the MIDI
        """
        voices = []
        num_notes_per_voice = []
        instrument_ids = []

        for instrument in midi.instruments:
            try:
                instrument_id = self.instrument_to_id[instrument.program]
            except KeyError:
                raise KeyError(
                    f"Instrument {instrument.program} has not been registered in NESDataset.instruments_id. "
                    f"As of now the registered instruments are {', '.join(self.instruments_ids.keys())} "
                    f"whereas the instruments of the current MIDI file are {', '.join([i.program for i in midi.instruments])}."
                )
            voice = []

            # add notes in the requested format
            for n in instrument.notes:
                pitch = 128 * instrument_id + n.pitch
                assert pitch > 0, str(n)
                # add the note to the list of notes
                voice.append([pitch, n.duration, n.velocity, n.start])

                # associate to the pitch a unique id if not done yet
                for p in range(max(128 * instrument_id, pitch - 6),
                               min(128 * (instrument_id + 1), pitch + 6)):
                    _ = self.pitches_dict[p]

            # ensure that notes are sorted by increasing starts
            voice.sort(key=lambda n: n[3])
            # add the voice to the list of voices
            voices.append(np.array(voice, dtype=np.float32))
            num_notes_per_voice.append(len(voice))
            instrument_ids.append(instrument_id)

        # pad voices # NOTE: padding value MUST ABSOLUTELY be -1 so that the trick in split_sequence works
        padded_voices = -np.ones(
            (max(num_notes_per_voice), len(self.id_to_instrument), 4),
            dtype=np.float32)
        for i, voice, num_notes in zip(instrument_ids, voices,
                                       num_notes_per_voice):
            padded_voices[:num_notes, i, :] = voice

        return padded_voices

    def generate_midi(self, notes_tensor):
        r"""
        """
        notes_per_instrument = defaultdict(list)

        if torch.is_tensor(notes_tensor):
            notes_tensor = notes_tensor.cpu().numpy()

        voice_pitch, duration, velocity, start = notes_tensor.T
        voice_pitch = voice_pitch.astype(np.int64)
        velocity = velocity.astype(np.int64)

        instrument_id, pitch = np.divmod(voice_pitch, 128)

        end = start + duration

        for i, v, p, s, e in zip(instrument_id, velocity, pitch, start, end):
            notes_per_instrument[i].append(
                pretty_midi.Note(velocity=v, pitch=p, start=s, end=e))

        midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=22050)

        for instrument_id, notes in notes_per_instrument.items():
            prog = self.id_to_instrument[instrument_id]
            instrument = pretty_midi.Instrument(program=prog,
                                                is_drum=(prog >= 112))

            instrument.notes = notes

            midi.instruments.append(instrument)

        ts = pretty_midi.TimeSignature(4, 4, 0)
        eos = pretty_midi.TimeSignature(1, 1, float(max(end) - min(start)))
        midi.time_signature_changes.extend([ts, eos])
        # print(midi.instruments[0].notes)
        # print(midi.instruments)
        # print(midi.time_signature_changes)
        # raise IndexError
        return midi


class SimpleNESDataset(Dataset):
    def __init__(self, phase='train', voices=None, sequences_size=None):
        self.sequences_size = sequences_size
        # self.root = (Path(__file__) /
        #              '../../../../data/simplenesmdb_midi').resolve()
        self.root = (Path(__file__) /
                     '../../../../data/long_nsf_midi_dataset').resolve()
        self.voices = voices if voices is not None else [0, 1, 2, 3]

        self.train = phase == 'train'

        if not self.root.exists():
            raise ValueError(
                f"The dataset could not be found in this folder: {self.root}.\n"
                "Move it to that folder or create a link if you already have it elsewhere.\n"
                "Otherwise run ./DatasetManager/nes/init_nes_dataset.sh to download and extract it (72 MB)."
            )

        self.processed = self.root / 'processed'
        self.processed.mkdir(exist_ok=True)
        self.smallest_time_shift = 0.02
        self.time_table_duration = get_time_table_duration(
            self.smallest_time_shift)
        self.time_table_time_shift = get_time_table_duration(
            self.smallest_time_shift)
        # assign an id to each instrument
        self.instrument_to_id = {80: 0, 81: 1, 38: 2, 121: 3}
        self.instrument_to_name = {80: 'p1', 81: 'p2', 38: 'tr', 121: 'no'}
        self.id_to_instrument = {
            v: k
            for k, v in self.instrument_to_id.items()
        }

        # Loading index dicts
        self.dict_path = self.root / 'pitches.json'
        if self.dict_path.exists():
            with open(self.dict_path, 'r') as f:
                pitches_dict = json.load(f)
                # JSON only supports type string for dict keys
                self.pitches_dict = {
                    int(k): v
                    for k, v in pitches_dict.items()
                }

        else:
            self.pitches_dict = defaultdict(
                lambda: len(self.pitches_dict.keys()), {
                    -1: -1,
                    0: 0
                })  # enable automatic updating of keys

            shutil.rmtree(self.processed)

            self.processed.mkdir(exist_ok=True)

            # preprocessing
            print(f'Preprocessing {self}...')
            t0 = time.time()
            self.preprocess_dataset()
            t1 = time.time()
            d = t1 - t0
            print(
                'Done. Time elapsed:', '{:.0f} s.'.format(d)
                if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))

            self.pitches_dict = dict(
                self.pitches_dict)  # remove defaultdict behaviour
            with open(self.dict_path, 'w') as f:
                json.dump(self.pitches_dict, f)

        self.paths = list((self.processed / phase).glob('*.npy'))

        self.compute_index_dicts()
        
    # TODO fix names
    # alias
    @property
    def sequence_size(self):
        return self.sequences_size
    
    def _data_augmentation(self, score):
        """
        inplace

        Args:
            score is (num_events, num_channels)
        # channels: (pitch, duration, velocity, start_time!)
        """
                
        # velocities
        velocity_shift = random.randint(-1, 1)
        score[:, 2] = np.maximum(np.minimum(score[:, 2] + velocity_shift, [15]), [1])

        # pitch
        melodic_shift = random.randint(-6, 5)
        melodic_voices = score[:, 0][score[:, 0] < 3 * 128]
        # assumes the original midi notes lie between [6 and 122]
        score[:, 0][score[:, 0] < 3 * 128] = melodic_voices + melodic_shift        

        # 2. adjust the speed of the piece by a random percentage between +/- 5%
            
        time_speed = 1 + (np.random.random() - 0.5) / 5

        score[:, 1] *= time_speed
        score[:, 3] *= time_speed
        # TODO data augmentation, switch between voices?
        return score
        

    def __getitem__(self, idx):
        # This sampling is NOT uniform!
        path = self.paths[idx]
        # print(path)
        score = np.load(path, allow_pickle=False)
        # score is (num_events, num_channels)
        # channels: (pitch, duration, velocity, start_time!)
        ## data augmentation
        if self.train:
            score = self._data_augmentation(score)

        # Tokenize
        # replace midi pitches by an id so that every id between 0 and id_max are used
        score_dict = self.tokenize(score)
        if self.sequences_size is not None:
            score_dict = self.add_start_end_pad_and_slice(data_dict=score_dict, 
                                                          start_time=None)
        return score_dict

    # Old method
    # def pad_or_slice(self, data_dict):
    #     pitch = data_dict['pitch']
    #     length = pitch.size(0)

    #     if length < self.sequences_size:
    #         pad_length = self.sequences_size - length

    #         for feature in data_dict:
    #             data_dict[feature] = torch.cat([
    #                 data_dict[feature],
    #                 torch.LongTensor([self.value2index[feature][END_SYMBOL]]),
    #                 torch.LongTensor([self.value2index[feature][PAD_SYMBOL]
    #                                   ]).repeat(pad_length - 1),
    #             ],
    #                                            dim=0)
    #     elif length > self.sequences_size:
    #         offset = random.randint(0, length - self.sequences_size - 1)
    #         for feature in data_dict:
    #             data_dict[feature] = data_dict[feature][offset:self.
    #                                                     sequences_size +
    #                                                     offset]
    #     else:
    #         pass
    #     return data_dict
    
    def add_start_end_pad_and_slice(self, data_dict, start_time=None):
        # TODO CHECK
        if start_time is None:
            # sample start time: between -sequences_size // 2 and sequences_size // 2
            # this is arbitrary but useful with prefix_decoder
            start_time = random.randint(
                - self.sequences_size // 2, self.sequences_size // 2
            )
        
        if start_time < 0:
            # we append PAD symbols and START
            num_symbols_to_add = -start_time
            for feature in data_dict:
                data_dict[feature] = torch.cat([                    
                    torch.LongTensor([self.value2index[feature][PAD_SYMBOL]
                                      ]).repeat(num_symbols_to_add - 1),
                    torch.LongTensor([self.value2index[feature][START_SYMBOL]]),
                    data_dict[feature]
                ],
                                               dim=0)
        
        # Same for the ending
        pitch = data_dict['pitch']
        length = pitch.size(0)
        
        # if we need to pad
        if length < self.sequences_size:
            pad_length = self.sequences_size - length
            for feature in data_dict:
                data_dict[feature] = torch.cat([
                    data_dict[feature],
                    torch.LongTensor([self.value2index[feature][END_SYMBOL]]),
                    torch.LongTensor([self.value2index[feature][PAD_SYMBOL]
                                      ]).repeat(pad_length - 1),
                ],
                                               dim=0)
        else:
            # we slice            
            data_dict = {
                k: v[:self.sequences_size]
                         for k, v in data_dict.items()
            }
        return data_dict
    


            
        for feature in list_of_features:
                list_of_features[feature] = torch.cat([
                    list_of_features[feature],
                    torch.LongTensor([self.value2index[feature][END_SYMBOL]]),
                    torch.LongTensor([self.value2index[feature][PAD_SYMBOL]
                                      ]).repeat(pad_length - 1),
                ],
                                               dim=0)

    def compute_index_dicts(self):

        self.index2value = {}
        self.value2index = {}
        #  Index 2 value
        for feat_name in ['pitch', 'duration', 'velocity', 'time_shift']:
            index2value = {}
            value2index = {}
            index = 0

            if feat_name == 'time_shift':
                values = self.time_table_time_shift
            elif feat_name == 'duration':
                values = self.time_table_duration[1:]
            elif feat_name == 'pitch':
                values = self.pitches_dict
            elif feat_name == 'velocity':
                values = list(range(16))
            else:
                raise Exception

            for value in values:
                index2value[index] = value
                value2index[value] = index
                index += 1

            # Pad
            index2value[index] = PAD_SYMBOL
            value2index[PAD_SYMBOL] = index
            index += 1
            # Start
            index2value[index] = START_SYMBOL
            value2index[START_SYMBOL] = index
            index += 1
            # End
            index2value[index] = END_SYMBOL
            value2index[END_SYMBOL] = index
            index += 1

            self.index2value[feat_name] = index2value
            self.value2index[feat_name] = value2index

    def tokenize(self, score):
        """

        Args:
            score (length, num_channels): 
            channels are pitch, duration, velocity, start_time            
        """

        pitch = np.vectorize(self.pitches_dict.__getitem__)(score[:, 0])
        pitch = torch.from_numpy(pitch).long()

        time = torch.from_numpy(score[:, 3])
        time_shift = torch.cat(
            [time[1:] - time[:-1], torch.zeros((1, ))], dim=0)
        time_shift = [
            self.value2index['time_shift'][find_nearest_value(
                self.time_table_time_shift, e.item())] for e in time_shift
        ]

        duration = [
            self.value2index['duration'][max(
                find_nearest_value(self.time_table_duration, e.item()),
                self.smallest_time_shift)] for e in score[:, 1]
        ]

        velocity = [
            self.value2index['velocity'][e.item()] for e in score[:, 2]
        ]

        return dict(
            pitch=pitch.long(),
            time_shift=torch.LongTensor(time_shift),
            duration=torch.LongTensor(duration),
            velocity=torch.LongTensor(velocity),
        )

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        return f'<NESDataset(dir={self.processed})>'

    def __str__(self):
        return 'NES-MDB Dataset'

    # PREPROCESSING

    def preprocess_dataset(self):
        r"""Recursively explore the dataset and generate for each MIDI file found
        a numpy array containing its data
        """
        bar = tqdm(sorted(list(self.root.glob('**/*.mid'))), leave=False)
        for src_file in bar:
            fname = src_file.name
            bar.set_description(fname[:29] +
                                '...' if len(fname) > 32 else fname +
                                (32 - len(fname)) * ' ')

            target_dir = self.processed / src_file.relative_to(
                self.root).parent
            target_name = src_file.stem + '.npy'
            target_file = target_dir / target_name

            # skip already preprocessed files
            if target_file.exists():
                continue

            target_dir.mkdir(exist_ok=True)

            # compute musical events as tensor
            try:
                midi = pretty_midi.PrettyMIDI(str(src_file))
                tensor = self.parse_midi(midi)
            except ValueError:
                warnings.warn(
                    f"Some error occurred during the preprocessing of {src_file}"
                )

            # save the computed tensor
            np.save(target_file, tensor, allow_pickle=False)

    def parse_midi(self, midi):
        r"""Read a MIDI file and return all notes in it in a numpy.ndarray

        Arguments:
            midi (pretty_midi.PrettyMIDI): MIDI file to be read

        Returns:
            numpy.ndarray: padded content of the MIDI
        """
        seq = []

        for instrument in midi.instruments:
            try:
                instrument_id = self.instrument_to_id[instrument.program]
            except KeyError:
                raise KeyError(
                    f"Instrument {instrument.program} has not been registered in NESDataset.instruments_id. "
                    f"As of now the registered instruments are {', '.join(self.instruments_ids.keys())} "
                    f"whereas the instruments of the current MIDI file are {', '.join([i.program for i in midi.instruments])}."
                )

            # add notes in the requested format
            for n in instrument.notes:
                pitch = 128 * instrument_id + n.pitch
                assert pitch > 0, str(n)
                # add the note to the list of notes
                seq.append([pitch, n.duration, n.velocity, n.start])

                # associate to the pitch a unique id if not done yet
                for p in range(max(128 * instrument_id, pitch - 6),
                               min(128 * (instrument_id + 1), pitch + 6)):
                    _ = self.pitches_dict[p]

        # ensure that notes are sorted by increasing starts and increasing pitch
        seq.sort(key=lambda n: (n[3], n[0]))

        return seq

    def detokenize(self, tensor_dict):
        tensor_dict_detokenized = {}
        
        # remove END and XX symbols
        end_symbol_location = np.inf
        for i, p in enumerate(tensor_dict['pitch']):
            if self.index2value['pitch'][p] in [END_SYMBOL]:
                end_symbol_location = i
                break                
            
        start_symbol_location = -1
        for i, p in enumerate(tensor_dict['pitch']):
            if self.index2value['pitch'][p] in [START_SYMBOL]:
                start_symbol_location = i                                
        
        for k, v in tensor_dict.items():
            channel = []
            for i, n in enumerate(v):
                print(f'{k} : {n} : {self.index2value[k][n]}')
                if start_symbol_location < i < end_symbol_location:
                    assert self.index2value[k][n] != PAD_SYMBOL
                    channel.append(self.index2value[k][n])

            tensor_dict_detokenized[k] = torch.Tensor(channel)
        print(start_symbol_location, end_symbol_location)
        # cast
        tensor_dict_detokenized['pitch'] = tensor_dict_detokenized[
            'pitch'].long().numpy()
        tensor_dict_detokenized['duration'] = tensor_dict_detokenized[
            'duration'].float().numpy()
        tensor_dict_detokenized['velocity'] = tensor_dict_detokenized[
            'velocity'].long().numpy()
        tensor_dict_detokenized['time_shift'] = tensor_dict_detokenized[
            'time_shift'].float().numpy()

        return tensor_dict_detokenized

    def generate_midi(self, tensor_dict):
        """
        tensor_dict: pitch, velocity, duration, time_shift
        where each is a LongTensor of size (num_events,)
        """
        # everything on cpu
        tensor_dict = {
            k: v.detach().cpu().numpy()
            for k, v in tensor_dict.items()
        }

        # detokenize
        tensor_dict = self.detokenize(tensor_dict)
        # TODO add zero

        start = np.concatenate([
            np.zeros(1),
            tensor_dict['time_shift'].cumsum(0)[:-1]], axis=0)
        end = start + tensor_dict['duration']
        pitch = tensor_dict['pitch']
        velocity = tensor_dict['velocity']

        midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=22050)

        notes_per_instrument = {
            i: []
            for i in range(len(self.id_to_instrument))
        }

        for p, v, s, e in zip(pitch, velocity, start, end):
            instrument_id, note_pitch = np.divmod(p, 128)

            notes_per_instrument[instrument_id].append(
                pretty_midi.Note(velocity=v, pitch=note_pitch, start=s, end=e))

        for instrument_id, notes in notes_per_instrument.items():
            prog = self.id_to_instrument[instrument_id]
            instrument_name = self.instrument_to_name[prog]
            instrument = pretty_midi.Instrument(program=prog,
                                                is_drum=(prog >= 112),
                                                name=instrument_name)

            instrument.notes = notes

            midi.instruments.append(instrument)

        ts = pretty_midi.TimeSignature(4, 4, 0)
        eos = pretty_midi.TimeSignature(1, 1, float(max(end) - min(start)))
        midi.time_signature_changes.extend([ts, eos])
        # print(midi.instruments[0].notes)
        # print(midi.instruments)
        # print(midi.time_signature_changes)
        # raise IndexError
        return midi
    
    

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

        train_dl = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val = SimpleNESDataset(phase='valid',
                               voices=self.voices,
                               sequences_size=self.sequences_size)
        val_dl = torch.utils.data.DataLoader(
            val,
            batch_size=batch_size,
            shuffle=shuffle_val,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        test_dl = torch.utils.data.DataLoader(
            val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return {'train': train_dl, 'val': val_dl, 'test': test_dl}

    def timeshift_indices_to_elapsed_time(self, timeshift_indices,
                                          smallest_time_shift):
        """
        Reverse operation than tokenize using get_time_table_duration
        """
        # TODO write proper test
        # WARNING any change here must be done in
        # get_time_table_duration for consistency
        y = torch.zeros_like(timeshift_indices).float()
        x = timeshift_indices

        # short time shifts
        num_short_time_shifts = int(1 / smallest_time_shift)
        y[x < num_short_time_shifts] = x[
            x < num_short_time_shifts].float() * smallest_time_shift

        # medium time shifts
        num_medium_time_shifts = int((5. - 1.) / (5.0 * smallest_time_shift))

        medium_mask = torch.logical_and(
            num_short_time_shifts <= x,
            x < num_short_time_shifts + num_medium_time_shifts)
        y[medium_mask] = 1. + (x[medium_mask] - num_short_time_shifts
                               ).float() * 5.0 * smallest_time_shift

        num_long_time_shifts = int((20. - 5.) / (50. * smallest_time_shift))

        long_mask = torch.logical_and(
            num_short_time_shifts + num_medium_time_shifts <= x,
            x < num_short_time_shifts + num_medium_time_shifts +
            num_long_time_shifts)
        y[long_mask] = 5. + (
            x[long_mask] - num_short_time_shifts -
            num_medium_time_shifts).float() * 50 * smallest_time_shift
        # if not (x <= (num_short_time_shifts + num_medium_time_shifts +
        #              num_long_time_shifts)).byte().all():
        #     print(x.data)
        # assert (x <= (num_short_time_shifts + num_medium_time_shifts +
        #              num_long_time_shifts)).byte().all()
        return y


if __name__ == '__main__':
    dataset = SimpleNESDataset(voices=(0, 1, 2, 3), phase='train')
    for i in range(10000):
        e = dataset[i]
        midi = dataset.generate_midi(e)
        midi.write('test.mid')
        exit()
