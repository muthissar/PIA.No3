from fractions import Fraction

import music21
from music21.chord_symbols.jazz_chords import JazzChord

import torch
from bson import ObjectId

import numpy as np
from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, standard_name, \
    standard_note, PAD_SYMBOL
from DatasetManager.lsdb.LsdbMongo import LsdbMongo
from DatasetManager.lsdb.lsdb_data_helpers import NC, notes_and_chords, leadsheet_on_ticks
from DatasetManager.music_dataset import MusicDataset
from DatasetManager.lsdb.lsdb_exceptions import *
from tqdm import tqdm


class LsdbDataset(MusicDataset):
    def __init__(self, corpus_it_gen,
                 name,
                 sequences_size,
                 cache_dir):
        """

        :param corpus_it_gn:
        :param sequences_size: in beats
        """
        super(LsdbDataset, self).__init__(cache_dir=cache_dir)
        self.name = name
        self.tick_values = [0,
                            Fraction(1, 4),
                            Fraction(1, 3),
                            Fraction(1, 2),
                            Fraction(2, 3),
                            Fraction(3, 4)]
        self.tick_durations = self.compute_tick_durations()
        self.number_of_beats = 4
        self.num_voices = 3
        self.NOTES = 0
        self.CHORD_ROOT = 1
        self.CHORD_NAME = 2
        self.corpus_it_gen = corpus_it_gen
        self.sequences_size = sequences_size
        self.subdivision = len(self.tick_values)
        self.pitch_range = [55, 84]
        self.init_index_dicts()

    def __repr__(self):
        # TODO
        return f'LsdbDataset(' \
               f'{self.name},' \
               f'{self.sequences_size})'

    def iterator_gen(self):
        return (score
                for score in self.corpus_it_gen()
                )

    def compute_tick_durations(self):
        diff = [n - p
                for n, p in zip(self.tick_values[1:], self.tick_values[:-1])]
        diff = diff + [1 - self.tick_values[-1]]
        return diff

    def transposed_score_and_metadata_tensors(self,
                                              score: music21.stream.Score,
                                              interval: music21.interval.Interval
                                              ):
        try:
            leadsheet_transposed = score.transpose(interval)
        except ValueError as e:
            raise LeadsheetParsingException(f'Leadsheet {leadsheet.metadata.title} '
                                            f'not properly formatted')
        return leadsheet_transposed

    def notes_to_lead_tensor(self, notes,
                             length: int,
                             update_dicts: bool = False):
        eps = 1e-4

        # LEAD
        j = 0
        i = 0
        t = np.zeros((length, 2))
        is_articulated = True
        num_notes = len(notes)
        current_tick = 0

        note2index = self.symbol2index_dicts[self.NOTES]
        index2note = self.index2symbol_dicts[self.NOTES]
        while i < length:
            # update dicts when creating the dataset
            note_name = standard_name(notes[j])
            if update_dicts and note_name not in note2index:
                new_index = len(note2index)
                note2index[note_name] = new_index
                index2note[new_index] = note_name

            note_index = note2index[note_name]
            if j < num_notes - 1:
                if notes[j + 1].offset > current_tick + eps:
                    t[i, :] = [note_index,
                               is_articulated]
                    i += 1
                    current_tick += self.tick_durations[
                        (i - 1) % len(self.tick_values)]
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [note_index,
                           is_articulated]
                i += 1
                is_articulated = False
        lead = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * note2index[SLUR_SYMBOL]
        lead_tensor = torch.from_numpy(lead).long()[None, :]

        return lead_tensor

    def chords_to_roots_and_types_tensors(self, chords, length,
                                          update_dicts: bool = False):
        # CHORDS
        j = 0
        i = 0
        t = np.zeros((length, 2))
        u = np.zeros((length, 2))
        is_articulated = True
        num_chords = len(chords)
        chordroot2index = self.symbol2index_dicts[self.CHORD_ROOT]
        index2chordroot = self.index2symbol_dicts[self.CHORD_ROOT]
        chordname2index = self.symbol2index_dicts[self.CHORD_NAME]
        index2chordname = self.index2symbol_dicts[self.CHORD_NAME]
        while i < length:
            # check if JazzChord
            if isinstance(chords[j], JazzChord):
                # update dicts when creating the dataset
                chord_root = standard_name(chords[j]).split(',')[0]
                chord_name = chords[j].chord_name
                if update_dicts and chord_root not in chordroot2index:
                    new_index = len(chordroot2index)
                    chordroot2index[chord_root] = new_index
                    index2chordroot[new_index] = chord_root
                chord_root_index = chordroot2index[chord_root]
                if update_dicts and chord_name not in chordname2index:
                    new_index = len(chordname2index)
                    chordname2index[chord_name] = new_index
                    index2chordname[new_index] = chord_name
                chord_name_index = chordname2index[chord_name]
            elif isinstance(chords[j], music21.expressions.TextExpression):
                content = chords[j].content
                if update_dicts and content not in chordroot2index:
                    new_index = len(chordroot2index)
                    chordroot2index[content] = new_index
                    index2chordroot[new_index] = content
                if update_dicts and content not in chordname2index:
                    new_index = len(chordname2index)
                    chordname2index[content] = new_index
                    index2chordname[new_index] = content
                chord_root_index = chordroot2index[content]
                chord_name_index = chordname2index[content]

            if j < num_chords - 1:
                if chords[j + 1].offset > i:
                    t[i, :] = [chord_root_index,
                               is_articulated]
                    u[i, :] = [chord_name_index,
                               is_articulated]
                    i += 1
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [chord_root_index,
                           is_articulated]
                u[i, :] = [chord_name_index,
                           is_articulated]
                i += 1
                is_articulated = False

        # TODO no SLUR_SYMBOL for chords?!
        # seq = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * chordroot2index[SLUR_SYMBOL]
        seq = t[:, 0]
        chord_root_tensor = torch.from_numpy(seq).long()[None, :]
        seq = u[:, 0]
        # seq = u[:, 0] * u[:, 1] + (1 - u[:, 1]) * chordname2index[SLUR_SYMBOL]
        chord_name_tensor = torch.from_numpy(seq).long()[None, :]

        return chord_root_tensor, chord_name_tensor

    def get_score_tensor(self, leadsheet: music21.stream.Score,
                         update_dicts: bool = False):
        """

        :param leadsheet:
        :return: lead_tensor and chord_tensor
        """
        notes, chords = notes_and_chords(leadsheet)
        if not leadsheet_on_ticks(leadsheet, self.tick_values):
            raise LeadsheetParsingException(
                f'Leadsheet {leadsheet.metadata.title} has notes not on ticks')

        length = int(leadsheet.highestTime * self.subdivision)
        lead_tensor = self.notes_to_lead_tensor(notes, length, update_dicts)

        if len(chords) > 0:
            length = int(leadsheet.highestTime)
            chord_root_tensor, chord_types_tensor = (
                self.chords_to_roots_and_types_tensors(chords, length,
                                                       update_dicts)
                )
        else:
            chord_root_tensor = torch.Tensor()
            chord_types_tensor = torch.Tensor()

        return lead_tensor, chord_root_tensor, chord_types_tensor

    def get_metadata_tensor(self, score):
        """

        :param score: music21 score object
        :return: torch tensor, with the score representation
                 as a tensor
        """
        raise NotImplementedError

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        # todo check on chorale with Chord
        print('Making tensor dataset')
        # todo not useful?
        # self.compute_index_dicts()
        lead_tensor_dataset = []
        chord_root_tensor_dataset = []
        chord_name_tensor_dataset = []
        count = 0
        num_scores = sum(1 for x in self.corpus_it_gen())
        if num_scores == 0:
            print('No scores available in LeadSheetIteratorGenerator')
            raise RuntimeError
        for _, leadsheet in tqdm(enumerate(self.corpus_it_gen())):
            print('Entered: ', count)
            count += 1
            print(leadsheet.metadata.title)
            if not self.is_valid(leadsheet):
                continue
            try:
                possible_transpositions = self.all_transposition_intervals(leadsheet)
                for transposition_interval in possible_transpositions:
                    transposed_leadsheet = self.transposed_score_and_metadata_tensors(
                        leadsheet,
                        transposition_interval)
                    lead_tensor, chord_root_tensor, chord_name_tensor = self.get_score_tensor(
                        transposed_leadsheet, update_dicts=True
                    )
                    # lead
                    for offsetStart in range(-self.sequences_size + 1,
                                             int(transposed_leadsheet.highestTime)):
                        offsetEnd = offsetStart + self.sequences_size
                        local_lead_tensor = self.extract_score_tensor_with_padding(
                            tensor=lead_tensor,
                            start_tick=offsetStart * self.subdivision,
                            end_tick=offsetEnd * self.subdivision,
                            symbol2index=self.symbol2index_dicts[self.NOTES]
                        )
                        local_chord_root_tensor = self.extract_score_tensor_with_padding(
                            tensor=chord_root_tensor,
                            start_tick=offsetStart,
                            end_tick=offsetEnd,
                            symbol2index=self.symbol2index_dicts[self.CHORD_ROOT]
                        )
                        local_chord_name_tensor = self.extract_score_tensor_with_padding(
                            tensor=chord_name_tensor,
                            start_tick=offsetStart,
                            end_tick=offsetEnd,
                            symbol2index=self.symbol2index_dicts[self.CHORD_NAME]
                        )
                        # append and add batch dimension
                        # cast to int
                        lead_tensor_dataset.append(
                            local_lead_tensor.int())
                        chord_root_tensor_dataset.append(
                            local_chord_root_tensor.int()
                        )
                        chord_name_tensor_dataset.append(
                            local_chord_name_tensor.int()
                        )
            except LeadsheetParsingException as e:
                print(e)

        lead_tensor_dataset = torch.cat(lead_tensor_dataset, 0)
        chord_root_tensor_dataset = torch.cat(chord_root_tensor_dataset, 0)
        chord_name_tensor_dataset = torch.cat(chord_name_tensor_dataset, 0)
        dataset = TensorDataset(lead_tensor_dataset,
                                chord_root_tensor_dataset,
                                chord_name_tensor_dataset
                                )

        print(f'Sizes: {lead_tensor_dataset.size()},'
              f' {chord_root_tensor_dataset.size()},'
              f' {chord_name_tensor_dataset.size()}'
              )
        return dataset

    def contains_notes_and_chords(self, leadsheet):
        notes_and_rests, chords = notes_and_chords(leadsheet)
        notes = [n.pitch.midi for n in notes_and_rests if n.isNote]
        return len(notes) > 0 and len(chords) > 0

    def all_transposition_intervals(self, leadsheet):
        min_pitch, max_pitch = self.leadsheet_range(leadsheet)
        min_pitch_corpus, max_pitch_corpus = self.pitch_range

        min_transposition = min_pitch_corpus - min_pitch
        max_transposition = max_pitch_corpus - max_pitch

        transpositions = []
        for semi_tone in range(min_transposition, max_transposition + 1):
            interval_type, interval_nature = music21.interval.convertSemitoneToSpecifierGeneric(
                semi_tone)
            transposition_interval = music21.interval.Interval(
                str(interval_nature) + interval_type)
            transpositions.append(transposition_interval)

        return transpositions

    def extract_score_tensor_with_padding(self,
                                          tensor,
                                          start_tick,
                                          end_tick,
                                          symbol2index):
        """

        :param tensor: (batch_size, length)
        :param start_tick:
        :param end_tick:
        :param symbol2index:
        :return: (batch_size, end_tick - start_tick)
        """
        assert start_tick < end_tick
        assert end_tick > 0
        batch_size, length = tensor.size()

        padded_tensor = []
        if start_tick < 0:
            start_symbols = np.array([symbol2index[PAD_SYMBOL]])
            start_symbols = torch.from_numpy(start_symbols).long().clone()
            start_symbols = start_symbols.repeat(batch_size, -start_tick)
            start_symbols[:, -1] = symbol2index[START_SYMBOL]
            padded_tensor.append(start_symbols)

        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length

        padded_tensor.append(tensor[:, slice_start: slice_end])

        if end_tick > length:
            end_symbols = np.array([symbol2index[PAD_SYMBOL]])
            end_symbols = torch.from_numpy(end_symbols).long().clone()
            end_symbols = end_symbols.repeat(batch_size, end_tick - length)
            end_symbols[:, 0] = symbol2index[END_SYMBOL]
            padded_tensor.append(end_symbols)

        padded_tensor = torch.cat(padded_tensor, 1)
        return padded_tensor

    def extract_metadata_with_padding(self,
                                      tensor_metadata,
                                      start_tick,
                                      end_tick):
        """

        :param tensor_metadata: torch tensor containing metadata
        :param start_tick:
        :param end_tick:
        :return:
        """
        raise NotImplementedError

    def is_lead(self, voice_id):
        return voice_id == self.NOTES

    def is_chord_root(self, voice_id):
        return voice_id == self.CHORD_ROOT

    def is_chord_name(self, voice_id):
        return voice_id == self.CHORD_NAME

    def init_index_dicts(self):
        print('Initialize index_dicts')
        self.index2symbol_dicts = [
            {} for _ in range(self.num_voices)
        ]
        self.symbol2index_dicts = [
            {} for _ in range(self.num_voices)
        ]

        # create and add additional symbols
        note_sets = [set() for _ in range(self.num_voices)]
        for note_set in note_sets:
            note_set.add(SLUR_SYMBOL)
            note_set.add(START_SYMBOL)
            note_set.add(END_SYMBOL)
            note_set.add(PAD_SYMBOL)

        # create tables
        for note_set, index2note, note2index in zip(note_sets,
                                                    self.index2symbol_dicts,
                                                    self.symbol2index_dicts):
            for note_index, note in enumerate(note_set):
                index2note.update({note_index: note})
                note2index.update({note: note_index})

    # Unused
    # def compute_index_dicts(self):
    #     print('Computing index dicts')
    #     self.index2symbol_dicts = [
    #         {} for _ in range(self.num_voices)
    #     ]
    #     self.symbol2index_dicts = [
    #         {} for _ in range(self.num_voices)
    #     ]
    #
    #     # create and add additional symbols
    #     note_sets = [set() for _ in range(self.num_voices)]
    #     for note_set in note_sets:
    #         note_set.add(SLUR_SYMBOL)
    #         note_set.add(START_SYMBOL)
    #         note_set.add(END_SYMBOL)
    #         note_set.add(PAD_SYMBOL)
    #
    #     # get all notes
    #     for leadsheet in tqdm(self.corpus_it_gen()):
    #         if self.is_in_range(leadsheet):
    #             # part is either lead or chords as lists
    #             for part_id, part in enumerate(notes_and_chords(leadsheet)):
    #                 for n in part:
    #                     note_sets[part_id].add(standard_name(n))
    #
    #     # create tables
    #     for note_set, index2note, note2index in zip(note_sets,
    #                                                 self.index2symbol_dicts,
    #                                                 self.symbol2index_dicts):
    #         for note_index, note in enumerate(note_set):
    #             index2note.update({note_index: note})
    #             note2index.update({note: note_index})

    #
    #
    # def compute_lsdb_chord_dicts(self):
    #     # TODO must be created from xml folder
    #     # Search LSDB for chord names
    #     with LsdbMongo() as mongo_client:
    #         db = mongo_client.get_db()
    #         modes = db.modes
    #         cursor_modes = modes.find({})
    #         chord2notes = {}  # Chord to notes dictionary
    #         notes2chord = {}  # Notes to chord dictionary
    #         for chord in cursor_modes:
    #             notes = []
    #             # Remove white spaces from notes string
    #             for note in re.compile("\s*,\s*").split(chord["chordNotes"]):
    #                 notes.append(note)
    #             notes = tuple(notes)
    #
    #             # Enter entries in dictionaries
    #             chord2notes[chord['mode']] = notes
    #             if notes in notes2chord:
    #                 notes2chord[notes] = notes2chord[notes] + [chord["mode"]]
    #             else:
    #                 notes2chord[notes] = [chord["mode"]]
    #
    #         self.correct_chord_dicts(chord2notes, notes2chord)
    #
    #         return chord2notes, notes2chord
    #
    # def correct_chord_dicts(self, chord2notes, notes2chord):
    #     """
    #     Modifies chord2notes and notes2chord in place
    #     to correct errors in LSDB modes (dict of chord symbols with notes)
    #     :param chord2notes:
    #     :param notes2chord:
    #     :return:
    #     """
    #     # Add missing chords
    #     # b5
    #     notes2chord[('C4', 'E4', 'Gb4')] = notes2chord[('C4', 'E4', 'Gb4')] + ['b5']
    #     chord2notes['b5'] = ('C4', 'E4', 'Gb4')
    #     # b9#5
    #     notes2chord[('C4', 'E4', 'G#4', 'Bb4', 'D#5')] = 'b9#b'
    #     chord2notes['b9#5'] = ('C4', 'E4', 'G#4', 'Bb4', 'D#5')
    #     # 7#5#11 is WRONG in the database
    #     # C4 F4 G#4 B-4 D5 instead  of C4 E4 G#4 B-4 D5
    #     notes2chord[('C4', 'E4', 'G#4', 'Bb4', 'F#5')] = '7#5#11'
    #     chord2notes['7#5#11'] = ('C4', 'E4', 'G#4', 'Bb4', 'F#5')
    #
    # # F#7#9#11 is WRONG in the database

    def test(self):
        with LsdbMongo() as client:
            db = client.get_db()
            leadsheets = db.leadsheets.find(
                {'_id': ObjectId('5193841a58e3383974000079')})
            leadsheet = next(leadsheets)
            print(leadsheet['title'])
            score = self.leadsheet_to_music21(leadsheet)
            score.show()

    def is_in_range(self, leadsheet):
        min_pitch, max_pitch = self.leadsheet_range(leadsheet)
        return (min_pitch >= self.pitch_range[0]
                and max_pitch <= self.pitch_range[1])

    def is_valid(self, leadsheet):
        return (self.contains_notes_and_chords(leadsheet)
                and
                self.is_in_range(leadsheet)
                )

    def leadsheet_range(self, leadsheet):
        notes, chords = notes_and_chords(leadsheet)
        pitches = [n.pitch.midi for n in notes if n.isNote]
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        return min_pitch, max_pitch

    def empty_score_tensor(self, score_length):
        """

        :param score_length: int, length of the score in ticks
        :return: torch long tensor, initialized with start indices
        """
        raise NotImplementedError

    def random_score_tensor(self, score_length):
        """

        :param score_length: int, length of the score in ticks
        :return: torch long tensor, initialized with random indices
        """
        lead_tensor = np.random.randint(len(self.symbol2index_dicts[self.NOTES]),
                                        size=score_length * self.subdivision)
        chord_roots_tensor = np.random.randint(len(self.symbol2index_dicts[self.CHORD_ROOT]),
                                               size=score_length)
        chord_types_tensor = np.random.randint(len(self.symbol2index_dicts[self.CHORD_NAME]),
                                               size=score_length)
        lead_tensor = torch.from_numpy(lead_tensor).long()
        chord_roots_tensor = torch.from_numpy(chord_roots_tensor).long()
        chord_types_tensor = torch.from_numpy(chord_types_tensor).long()
        return lead_tensor, chord_roots_tensor, chord_types_tensor

    def get_jazzchord_from_index(self, chord_root_index, chord_name_index):
        '''
        Returns a JazzChord based on the chord_idx
        '''
        index2chordroot = self.index2symbol_dicts[self.CHORD_ROOT]
        index2chordname = self.index2symbol_dicts[self.CHORD_NAME]
        chord_root = index2chordroot[chord_root_index]
        chord_name = index2chordname[chord_name_index]
        root_str = chord_root.replace('b', '-')
        root_pitch = music21.pitch.Pitch(root_str)
        chord_id = JazzChord.get_chord_id_from_name_str(chord_name)
        jazz_chord = JazzChord(chord_id, root_pitch)
        return jazz_chord

    def tensor_to_score(self, tensor_score, tensor_chords,
                        realize_chords=False, add_chord_symbols=False):
        """
        Converts given leadsheet as tensor_lead and tensor_chords
        to a true music21 score
        :param tensor_lead:
        :param tensor_chords:
        :return:
        """
        score = music21.stream.Score()
        part = music21.stream.Part()

        # LEAD
        dur = 0
        f = music21.note.Rest()
        tensor_score_np = tensor_score.numpy().flatten()
        slur_index = self.symbol2index_dicts[self.NOTES][SLUR_SYMBOL]
        for tick_index, note_index in enumerate(tensor_score_np):
            note_index = note_index.item()
            # if it is a played note
            if not note_index == slur_index:
                # add previous note
                if dur > 0:
                    f.duration = music21.duration.Duration(dur)
                    part.append(f)
                # TODO two types of tick_durations
                dur = self.tick_durations[tick_index % self.subdivision]
                f = standard_note(self.index2symbol_dicts[self.NOTES][note_index])
            else:
                dur += self.tick_durations[tick_index % self.subdivision]
        # add last note
        f.duration = music21.duration.Duration(dur)
        part.append(f)

        # CHORD SYMBOLS
        if add_chord_symbols:
            # index2chordroot = self.index2symbol_dicts[self.CHORD_ROOT]
            chordroot2index = self.symbol2index_dicts[self.CHORD_ROOT]
            # index2chordname = self.index2symbol_dicts[self.CHORD_NAME]
            # chordname2index = self.symbol2index_dicts[self.CHORD_NAME]
            start_index = chordroot2index[START_SYMBOL]
            end_index = chordroot2index[END_SYMBOL]
            slur_index = chordroot2index[SLUR_SYMBOL]
            pad_index = chordroot2index[PAD_SYMBOL]
            nc_index = chordroot2index[NC]

            chordtype2index = self.symbol2index_dicts[self.CHORD_NAME]
            # index2chordname = self.index2symbol_dicts[self.CHORD_NAME]
            # chordname2index = self.symbol2index_dicts[self.CHORD_NAME]
            type_start_index = chordtype2index[START_SYMBOL]
            type_end_index = chordtype2index[END_SYMBOL]
            type_slur_index = chordtype2index[SLUR_SYMBOL]
            type_pad_index = chordtype2index[PAD_SYMBOL]
            type_nc_index = chordtype2index[NC]

            tensor_chords_root, tensor_chords_name = tensor_chords
            tensor_chords_root_np = tensor_chords_root.numpy().flatten()
            tensor_chords_name_np = tensor_chords_name.numpy().flatten()
            for beat_index, (chord_root_index, chord_type_index) \
                    in enumerate(
                zip(
                    tensor_chords_root_np,
                    tensor_chords_name_np
                )
            ):
                chord_root_index = chord_root_index.item()
                chord_type_index = chord_type_index.item()
                # if it is a played chord
                # todo check also chord_type_index!
                if (chord_root_index not in [slur_index,
                                             start_index,
                                             end_index,
                                             pad_index,
                                             nc_index]
                        and
                        chord_type_index not in [type_slur_index,
                                                 type_start_index,
                                                 type_end_index,
                                                 type_pad_index,
                                                 type_nc_index]
                ):
                    # add chord
                    jazz_chord = self.get_jazzchord_from_index(
                        chord_root_index,
                        chord_type_index
                    )
                    part.insert(beat_index, jazz_chord)

            score.append(part)
        else:
            score.append(part)

        if realize_chords:
            # index2chordroot = self.index2symbol_dicts[self.CHORD_ROOT]
            chordroot2index = self.symbol2index_dicts[self.CHORD_ROOT]
            start_index = chordroot2index[START_SYMBOL]
            end_index = chordroot2index[END_SYMBOL]
            slur_index = chordroot2index[SLUR_SYMBOL]
            pad_index = chordroot2index[PAD_SYMBOL]
            nc_index = chordroot2index[NC]
            chords_part = music21.stream.Part()
            dur = 0
            c = music21.note.Rest()
            tensor_chords_root, tensor_chords_name = tensor_chords
            tensor_chords_root_np = tensor_chords_root.numpy().flatten()
            tensor_chords_name_np = tensor_chords_name.numpy().flatten()
            for (beat_index,
                 (chord_root_index, chord_type_index)) \
                    in enumerate(
                zip(
                    tensor_chords_root_np,
                    tensor_chords_name_np
                )
            ):
                chord_root_index = chord_root_index.item()
                chord_type_index = chord_type_index.item()
                # if it is a played note
                if chord_root_index not in [slur_index,
                                            start_index,
                                            end_index,
                                            pad_index,
                                            nc_index]:
                    # add previous note
                    if dur > 0:
                        c.duration = music21.duration.Duration(dur)
                        chords_part.append(c)
                    dur = 1
                    try:
                        jazz_chord = self.get_jazzchord_from_index(
                            chord_root_index,
                            chord_type_index
                        )
                        voicing_pitch_list = jazz_chord.get_pitchlist_from_chord()
                        c = music21.chord.Chord([
                            p.transpose(-12) for p in voicing_pitch_list
                        ])
                    except:
                        c = music21.note.Rest()
                else:
                    dur += 1
            # add last note
            c.duration = music21.duration.Duration(dur)
            chords_part.append(c)
            score.append(chords_part)

        return score

    def tensor_leadsheet_to_score_and_chord_list(self,
                                                 tensor_score,
                                                 tensor_chords,
                                                 add_chord_symbols=True,
                                                 realize_chords=False):
        """
        Converts leadsheet given as tensor_lead to a true music21 score
        and the chords as a list
        :param tensor_score:
        :param tensor_chords: tuple (tensor_chord_roots, tensor_chord_types)
        :return:
        """
        score = self.tensor_to_score(
            tensor_score,
            tensor_chords,
            realize_chords=realize_chords,
            add_chord_symbols=add_chord_symbols
        )

        # CHORDS LIST
        chord_list = []
        tensor_chords_root, tensor_chords_name = tensor_chords
        index2chordroot = self.index2symbol_dicts[self.CHORD_ROOT]
        index2chordname = self.index2symbol_dicts[self.CHORD_NAME]
        tensor_chords_root_np = tensor_chords_root.numpy().flatten()
        tensor_chords_name_np = tensor_chords_name.numpy().flatten()
        for chord_root_index, chord_name_index \
                in zip(tensor_chords_root_np,
                       tensor_chords_name_np
                       ):
            chord_root_index = chord_root_index.item()
            chord_name_index = chord_name_index.item()
            chord_desc = index2chordroot[chord_root_index] + \
                         index2chordname[chord_name_index]
            chord_list.append(chord_desc)
        return score, chord_list


if __name__ == '__main__':
    from DatasetManager.dataset_manager import DatasetManager

    dataset_manager = DatasetManager()
    leadsheet_dataset_kwargs = {
        'sequences_size': 64,
    }

    lsdb_dataset: LsdbDataset = dataset_manager.get_dataset(
        name='lsdb_test',
        **leadsheet_dataset_kwargs
    )
    dl, _, _ = lsdb_dataset.data_loaders(1)
    tensor_lead, tensor_chord_root, tensor_chord_name = next(dl.__iter__())
    print(tensor_lead[0].size(),
          tensor_chord_root[0].size(),
          tensor_chord_name[0].size())
    tensor_chord = (tensor_chord_root[0], tensor_chord_name[0])
    score, chord_list = lsdb_dataset.tensor_leadsheet_to_score_and_chord_list(
        tensor_lead[0],
        tensor_chord)
    score.show()
    print(chord_list)
    # leadsheet_path = '/home/ashis/Documents/AnticipationRNNFolkDataset/DatasetManager/DatasetManager/lsdb/xml/4_4_all/52nd Street Theme.xml'
    # leadsheet = music21.converter.parse(leadsheet_path)
    # lead, chord_root, chord_name = lsdb_dataset.get_score_tensor(leadsheet)
    # chord = (chord_root, chord_name)
    # score = lsdb_dataset.tensor_to_score(
    #    lead,
    #    chord,
    #    realize_chords=True
    # )
    # score.show()
