import os
import pickle
import re
import shutil

import music21
import numpy as np
import torch
from tqdm import tqdm

import DatasetManager
from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset
from DatasetManager.arrangement.arrangement_helper import shift_pr_along_pitch_axis, note_to_midiPitch, \
    score_to_pianoroll, new_events
from DatasetManager.helpers import REST_SYMBOL, SLUR_SYMBOL, END_SYMBOL, START_SYMBOL, \
    YES_SYMBOL, NO_SYMBOL, TIME_SHIFT, PAD_SYMBOL, STOP_SYMBOL

"""
Typical piano sequence:
p0 p1 TS p0 p1 p2 TS p0 STOP X X X X

If beginning: 
START p0 p1 TS p0 p1 p2 TS p0 STOP X X X

If end: 
p0 p1 TS p0 p1 p2 TS p0 END STOP X X X

"""


class ArrangementMidipianoDataset(ArrangementDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 corpus_it_gen_instru_range,
                 name,
                 mean_number_messages_per_time_frame,
                 subdivision,
                 sequence_size,
                 max_transposition,
                 integrate_discretization,
                 alignement_type,
                 transpose_to_sounding_pitch,
                 compute_statistics_flag):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        """
        velocity_quantization = 2
        super().__init__(corpus_it_gen=corpus_it_gen,
                         corpus_it_gen_instru_range=corpus_it_gen_instru_range,
                         name=name,
                         subdivision=subdivision,
                         sequence_size=sequence_size,
                         velocity_quantization=velocity_quantization,
                         max_transposition=max_transposition,
                         integrate_discretization=integrate_discretization,
                         alignement_type=alignement_type,
                         transpose_to_sounding_pitch=transpose_to_sounding_pitch,
                         compute_statistics_flag=compute_statistics_flag
                         )

        # Only piano is overwritten
        self.max_number_messages_piano = self.sequence_size * mean_number_messages_per_time_frame
        self.smallest_padding_length = self.max_number_messages_piano
        self.symbol2index_piano = {}
        self.index2symbol_piano = {}
        # Instruments presence
        self.instruments_presence2index = {}
        self.index2instruments_presence = {}
        self.instrument_presence_name2index = {}
        self.instrument_presence_index2name = {}

        self.precomputed_vectors_piano = {
            START_SYMBOL: None,
            END_SYMBOL: None,
            PAD_SYMBOL: None,
            REST_SYMBOL: None,
        }
        self.precomputed_vectors_orchestra = {
            START_SYMBOL: None,
            END_SYMBOL: None,
            PAD_SYMBOL: None,
            REST_SYMBOL: None,
        }

        self.precomputed_vectors_orchestra_instruments_presence = {
            PAD_SYMBOL: None
        }

        return

    def __repr__(self):
        name = f'ArrangementMidipianoDataset-' \
            f'{self.name}-' \
            f'{self.subdivision}-' \
            f'{self.sequence_size}-' \
            f'{self.max_transposition}'
        return name

    # def iterator_gen(self):

    # def iterator_gen_complementary(self):

    def load_index_dicts(self):
        dataset_manager_path = os.path.abspath(DatasetManager.__path__[0])
        index_dict_path = f'{dataset_manager_path}/dataset_cache/index_dicts/{type(self).__name__}.pkl'
        if not os.path.isfile(index_dict_path):
            print('Building index dictionnary. Might take some time')
            answer = None
            while answer not in ['y', 'n']:
                answer = input('Continue? Type y or n\n')
                if answer == 'y':
                    self.compute_index_dicts(index_dict_path)
                elif answer == 'n':
                    raise Exception('Aborting')

        with open(index_dict_path, 'rb') as ff:
            index_dicts = pickle.load(ff)

        #  Orchestra
        self.index2instrument = index_dicts['index2instrument']
        self.instrument2index = index_dicts['instrument2index']
        self.index2midi_pitch = index_dicts['index2midi_pitch']
        self.midi_pitch2index = index_dicts['midi_pitch2index']
        # Only piano is overwritten
        self.symbol2index_piano = index_dicts['symbol2index_piano']
        self.index2symbol_piano = index_dicts['index2symbol_piano']
        # Instruments presence
        self.instruments_presence2index = index_dicts['instruments_presence2index']
        self.index2instruments_presence = index_dicts['index2instruments_presence']
        self.instrument_presence_name2index = index_dicts['instrument_presence_name2index']
        self.instrument_presence_index2name = index_dicts['instrument_presence_index2name']
        # Dimensions
        self.number_instruments = index_dicts['number_instruments']
        self.instrument_presence_dim = index_dicts['instrument_presence_dim']
        # Misc
        self.observed_tessitura = index_dicts['observed_tessitura']

        ############################################################
        ############################################################
        orchestra_start_vector = []
        orchestra_end_vector = []
        orchestra_pad_vector = []
        orchestra_rest_vector = []
        orchestra_mask_vector = []
        for instru_ind, mapping in self.midi_pitch2index.items():
            orchestra_start_vector.append(mapping[START_SYMBOL])
            orchestra_pad_vector.append(mapping[PAD_SYMBOL])
            orchestra_end_vector.append(mapping[END_SYMBOL])
            orchestra_rest_vector.append(mapping[REST_SYMBOL])
        self.precomputed_vectors_orchestra[START_SYMBOL] = torch.from_numpy(np.asarray(orchestra_start_vector)).long()
        self.precomputed_vectors_orchestra[PAD_SYMBOL] = torch.from_numpy(np.asarray(orchestra_pad_vector)).long()
        self.precomputed_vectors_orchestra[END_SYMBOL] = torch.from_numpy(np.asarray(orchestra_end_vector)).long()
        self.precomputed_vectors_orchestra[REST_SYMBOL] = torch.from_numpy(np.asarray(orchestra_rest_vector)).long()
        #
        unknown_vector = np.ones((self.instrument_presence_dim)) * self.instruments_presence2index[PAD_SYMBOL]
        self.precomputed_vectors_orchestra_instruments_presence[PAD_SYMBOL] = \
            torch.from_numpy(unknown_vector).long()
        ###########################################################
        ############################################################
        return

    def compute_index_dicts(self, index_dict_path):
        if self.compute_tessitura:
            ############################################################
            ############################################################
            #  Mapping midi_pitch to token for each instrument

            # Hardcoded real life piano tessitura, so that we are not fucked at generation time if we see a score with out of range notes
            set_midiPitch_per_instrument = {'Piano': set(range(21, 108))}

            ############################################################
            # First pass over the database to create the mapping pitch <-> index for each instrument
            for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen())):

                if arr_pair is None:
                    continue

                # Compute pianoroll representations of score (more efficient than manipulating the music21 streams)
                # pianoroll_piano, onsets_piano, _ = score_to_pianoroll(arr_pair['Piano'], self.subdivision,
                #                                                       None,
                #                                                       self.instrument_grouping,
                #                                                       self.transpose_to_sounding_pitch)
                # pitch_set_this_track = set(np.where(np.sum(pianoroll_piano['Piano'], axis=0) > 0)[0])
                # set_midiPitch_per_instrument['Piano'] = set_midiPitch_per_instrument['Piano'].union(
                #     pitch_set_this_track)

                pianoroll_orchestra, onsets_orchestra, _ = score_to_pianoroll(score=arr_pair['Orchestra'],
                                                                              subdivision=self.subdivision,
                                                                              simplify_instrumentation=self.simplify_instrumentation,
                                                                              instrument_grouping=self.instrument_grouping,
                                                                              transpose_to_sounding_pitch=self.transpose_to_sounding_pitch,
                                                                              integrate_discretization=self.integrate_discretization,
                                                                              binarize=True)

                for instrument_name in pianoroll_orchestra:
                    if instrument_name not in set_midiPitch_per_instrument.keys():
                        set_midiPitch_per_instrument[instrument_name] = set()
                    pitch_set_this_track = set(np.where(np.sum(pianoroll_orchestra[instrument_name], axis=0) > 0)[0])
                    set_midiPitch_per_instrument[instrument_name] = set_midiPitch_per_instrument[instrument_name].union(
                        pitch_set_this_track)

            ############################################################
            # Potentially, we may want to also include ranges from an other database
            if self.corpus_it_gen_instru_range is not None:
                for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen_complementary())):

                    if arr_pair is None:
                        continue

                    pianoroll_orchestra, _, _ = score_to_pianoroll(
                        score=arr_pair['Orchestra'],
                        subdivision=self.subdivision,
                        simplify_instrumentation=self.simplify_instrumentation,
                        instrument_grouping=self.instrument_grouping,
                        transpose_to_sounding_pitch=self.transpose_to_sounding_pitch,
                        integrate_discretization=self.integrate_discretization,
                        binarize=True,
                    )

                    for instrument_name in pianoroll_orchestra:
                        if instrument_name not in set_midiPitch_per_instrument.keys():
                            set_midiPitch_per_instrument[instrument_name] = set()
                        pitch_set_this_track = set(
                            np.where(np.sum(pianoroll_orchestra[instrument_name], axis=0) > 0)[0])
                        set_midiPitch_per_instrument[instrument_name] = set_midiPitch_per_instrument[
                            instrument_name].union(
                            pitch_set_this_track)
        else:
            set_midiPitch_per_instrument = {}
            instrument_name_list = list(self.instrumentation.keys())
            instrument_name_list.append("Piano")
            for instrument_name in instrument_name_list:
                lowest_note, highest_note = self.reference_tessitura[instrument_name]
                lowest_pitch = note_to_midiPitch(lowest_note)
                highest_pitch = note_to_midiPitch(highest_note)
                set_pitches = set(range(lowest_pitch, highest_pitch + 1))
                set_midiPitch_per_instrument[instrument_name] = set_pitches

        ############################################################
        # Save statistics in a file
        if self.compute_statistics_flag:
            with open(f"{self.statistic_folder}/note_frequency_per_instrument", "w") as ff:
                for instrument_name, set_pitch_class in set_midiPitch_per_instrument.items():
                    ff.write(f"# {instrument_name}: \n")
                    for pc in set_pitch_class:
                        ff.write(f"   {pc}\n")

        ############################################################
        # Orchestra dicts
        midi_pitch2index_per_instrument = {}
        index2midi_pitch_per_instrument = {}
        for instrument_name, set_midiPitch in set_midiPitch_per_instrument.items():
            min_pitch = min(set_midiPitch)
            max_pitch = max(set_midiPitch)
            self.observed_tessitura[instrument_name] = {
                "min": min_pitch,
                "max": max_pitch
            }
            if instrument_name == "Piano":
                continue
            # Use range to avoid gaps in instruments tessitura (needed since we use
            # pitch transpositions as data augmentations
            list_midiPitch = sorted(list(range(min_pitch, max_pitch + 1)))
            midi_pitch2index_per_instrument[instrument_name] = {}
            index2midi_pitch_per_instrument[instrument_name] = {}
            for index, midi_pitch in enumerate(list_midiPitch):
                midi_pitch2index_per_instrument[instrument_name][midi_pitch] = index
                index2midi_pitch_per_instrument[instrument_name][index] = midi_pitch
            # Silence
            index += 1
            midi_pitch2index_per_instrument[instrument_name][REST_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = REST_SYMBOL
            #  Slur
            index += 1
            midi_pitch2index_per_instrument[instrument_name][SLUR_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = SLUR_SYMBOL
            # Pad
            index += 1
            midi_pitch2index_per_instrument[instrument_name][PAD_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = PAD_SYMBOL
            # Start
            index += 1
            midi_pitch2index_per_instrument[instrument_name][START_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = START_SYMBOL
            # End
            index += 1
            midi_pitch2index_per_instrument[instrument_name][END_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = END_SYMBOL

        # Print instruments avoided
        print("Instruments not used")
        for instrument_name in midi_pitch2index_per_instrument.keys():
            if self.instrumentation[instrument_name] == 0:
                print(f'# {instrument_name}')

        # Mapping instruments <-> indices
        index_counter = 0
        counter_instrument_presence = 0
        for instrument_name, number_instruments in self.instrumentation.items():
            if instrument_name == "Piano":
                continue
            #  Check if instrument appears in the dataset
            if instrument_name not in midi_pitch2index_per_instrument.keys():
                continue

            #  Don't use instruments which are assigned 0 voices
            if number_instruments == 0:
                continue

            self.instrument2index[instrument_name] = list(range(index_counter, index_counter + number_instruments))
            for ind in range(index_counter, index_counter + number_instruments):
                self.index2instrument[ind] = instrument_name
            index_counter += number_instruments

            self.instrument_presence_name2index[instrument_name] = counter_instrument_presence
            self.instrument_presence_index2name[counter_instrument_presence] = instrument_name
            counter_instrument_presence += 1

        # Mapping pitch <-> index per voice (that's the one we'll use, easier to manipulate when training)
        for instrument_name, instrument_indices in self.instrument2index.items():
            for instrument_index in instrument_indices:
                self.midi_pitch2index[instrument_index] = midi_pitch2index_per_instrument[instrument_name]
                self.index2midi_pitch[instrument_index] = index2midi_pitch_per_instrument[instrument_name]
        ############################################################
        ############################################################

        ############################################################
        ############################################################
        # Piano dicts
        min_pitch_piano = self.observed_tessitura['Piano']['min']
        max_pitch_piano = self.observed_tessitura['Piano']['max'] + 1
        index = 0
        for midi_pitch in range(min_pitch_piano, max_pitch_piano):
            symbol = str(midi_pitch) + '_on'
            self.symbol2index_piano[symbol] = index
            self.index2symbol_piano[index] = symbol
            index += 1
            symbol = str(midi_pitch) + '_off'
            self.symbol2index_piano[symbol] = index
            self.index2symbol_piano[index] = symbol
            index += 1
        # Time shift
        self.symbol2index_piano[TIME_SHIFT] = index
        self.index2symbol_piano[index] = TIME_SHIFT
        index += 1
        # Time shift
        self.symbol2index_piano[STOP_SYMBOL] = index
        self.index2symbol_piano[index] = STOP_SYMBOL
        index += 1
        # Padding
        self.symbol2index_piano[PAD_SYMBOL] = index
        self.index2symbol_piano[index] = PAD_SYMBOL
        index += 1
        # Start
        self.symbol2index_piano[START_SYMBOL] = index
        self.index2symbol_piano[index] = START_SYMBOL
        index += 1
        # End
        self.symbol2index_piano[END_SYMBOL] = index
        self.index2symbol_piano[index] = END_SYMBOL
        ############################################################
        ############################################################

        ############################################################
        ############################################################
        # Encoding for orchestra presence
        # Same mapping for all instruments
        #  Unknown symbol is used for dropout during training, and also when generating if you don't want to
        # hard constrain the presence/absence of a note
        self.instruments_presence2index = {
            NO_SYMBOL: 0,
            YES_SYMBOL: 1,
            PAD_SYMBOL: 2
        }
        self.index2instruments_presence = {}
        for k, v in self.instruments_presence2index.items():
            self.index2instruments_presence[v] = k
        ############################################################
        ############################################################

        self.number_instruments = len(self.midi_pitch2index)
        self.instrument_presence_dim = len(self.instrument_presence_index2name)

        #  Orchestra
        index_dicts = {
            'index2instrument': self.index2instrument,
            'instrument2index': self.instrument2index,
            'index2midi_pitch': self.index2midi_pitch,
            'midi_pitch2index': self.midi_pitch2index,
            'symbol2index_piano': self.symbol2index_piano,
            'index2symbol_piano': self.index2symbol_piano,
            'instruments_presence2index': self.instruments_presence2index,
            'index2instruments_presence': self.index2instruments_presence,
            'instrument_presence_name2index': self.instrument_presence_name2index,
            'instrument_presence_index2name': self.instrument_presence_index2name,
            'number_instruments': self.number_instruments,
            'instrument_presence_dim': self.instrument_presence_dim,
            'observed_tessitura': self.observed_tessitura
        }

        with open(index_dict_path, 'wb') as ff:
            pickle.dump(index_dicts, ff)

        return

    # def make_tensor_dataset(self, frame_orchestra=None):

    # def get_score_tensor(self, scores, offsets):

    # def transposed_score_and_metadata_tensors(self, score, semi_tone):

    # def get_metadata_tensor(self, score):

    # def score_to_list_pc(self, score, datatype):

    # def align_score(self, piano_score, orchestra_score):

    # def get_allowed_transpositions_from_pr(self, pr, frames, instrument_name):

    # def prepare_chunk_from_corresponding_frames(self, corresponding_frames):

    def transpose_loop_iteration(self, pianorolls_pair, onsets_pair, transposition_semi_tone,
                                 chunks_piano_indices, chunks_orchestra_indices,
                                 minimum_transposition_allowed, maximum_transposition_allowed,
                                 piano_tensor_dataset, orchestra_tensor_dataset,
                                 orchestra_instruments_presence_tensor_dataset,
                                 total_chunk_counter, too_many_instruments_frame, impossible_transposition):

        ############################################################
        # Transpose pianorolls
        this_pr_piano = shift_pr_along_pitch_axis(pianorolls_pair["Piano"], transposition_semi_tone)
        this_onsets_piano = shift_pr_along_pitch_axis(onsets_pair["Piano"], transposition_semi_tone)

        this_pr_orchestra = {}
        this_onsets_orchestra = {}
        for instrument_name in pianorolls_pair["Orchestra"].keys():
            # Pr
            pr = pianorolls_pair["Orchestra"][instrument_name]
            shifted_pr = shift_pr_along_pitch_axis(pr, transposition_semi_tone)
            this_pr_orchestra[instrument_name] = shifted_pr
            # Onsets
            onsets = onsets_pair["Orchestra"][instrument_name]
            shifted_onsets = shift_pr_along_pitch_axis(onsets, transposition_semi_tone)
            this_onsets_orchestra[instrument_name] = shifted_onsets
        ############################################################

        if minimum_transposition_allowed is None:
            if transposition_semi_tone != 0:
                raise Exception("Possible transpositions should be computed on non transposed pianorolls")
            # We have to construct the possible transpose
            build_allowed_transposition_flag = True
            minimum_transposition_allowed = []
            maximum_transposition_allowed = []
        else:
            build_allowed_transposition_flag = False

        for chunk_index in range(len(chunks_piano_indices)):
            this_chunk_piano_indices = chunks_piano_indices[chunk_index]
            this_chunk_orchestra_indices = chunks_orchestra_indices[chunk_index]
            avoid_this_chunk = False
            total_chunk_counter += 1

            ############################################################
            #  Get list of allowed transpositions
            if build_allowed_transposition_flag:
                min_transposition = -self.max_transposition
                max_transposition = self.max_transposition

                # Observe tessitura for each instrument for this chunk. Use non transposed pr of course
                this_min_transposition, this_max_transposition = \
                    self.get_allowed_transpositions_from_pr(this_pr_piano,
                                                            this_chunk_piano_indices,
                                                            "Piano")

                if (this_min_transposition is None) or (this_max_transposition is None):
                    min_transposition = None
                    max_transposition = None
                else:
                    min_transposition = max(this_min_transposition, min_transposition)
                    max_transposition = min(this_max_transposition, max_transposition)

                # Use reference tessitura or compute tessitura directly on the files ?
                if min_transposition is not None:
                    for instrument_name, pr in this_pr_orchestra.items():
                        this_min_transposition, this_max_transposition = \
                            self.get_allowed_transpositions_from_pr(pr,
                                                                    this_chunk_orchestra_indices,
                                                                    instrument_name)
                        if this_min_transposition is not None:  # If instrument not in this chunk, None was returned
                            min_transposition = max(this_min_transposition, min_transposition)
                            max_transposition = min(this_max_transposition, max_transposition)

                    this_minimum_transposition_allowed = min(0, min_transposition)
                    this_maximum_transposition_allowed = max(0, max_transposition)
                else:
                    this_minimum_transposition_allowed = None
                    this_maximum_transposition_allowed = None
                minimum_transposition_allowed.append(this_minimum_transposition_allowed)
                maximum_transposition_allowed.append(this_maximum_transposition_allowed)
            else:
                this_minimum_transposition_allowed = minimum_transposition_allowed[chunk_index]
                this_maximum_transposition_allowed = maximum_transposition_allowed[chunk_index]

            ############################################################
            #  Test if the transposition is possible
            if (this_maximum_transposition_allowed is None) or (this_minimum_transposition_allowed is None):
                impossible_transposition += 1
                continue
            if (this_minimum_transposition_allowed > transposition_semi_tone) \
                    or (this_maximum_transposition_allowed < transposition_semi_tone):
                impossible_transposition += 1
                continue

            ############################################################
            #  Extract representations
            local_piano_tensor = []
            local_orchestra_tensor = []
            local_orchestra_instruments_presence_tensor = []
            previous_notes_orchestra = None
            for index_frame in range(len(this_chunk_piano_indices)):
                if index_frame != 0:
                    previous_frame_piano = this_chunk_piano_indices[index_frame - 1]
                    if previous_frame_piano in [START_SYMBOL, END_SYMBOL]:
                        previous_frame_piano = None
                else:
                    previous_frame_piano = None
                frame_piano = this_chunk_piano_indices[index_frame]
                frame_orchestra = this_chunk_orchestra_indices[index_frame]

                if frame_orchestra in [START_SYMBOL, PAD_SYMBOL, END_SYMBOL]:
                    if frame_piano == END_SYMBOL:
                        #  Remove TS (after asserting there is one...)
                        assert local_piano_tensor[-1] == self.symbol2index_piano[TIME_SHIFT], "Weirdness"
                        local_piano_tensor.pop()
                        # Add end symbol, then the stop symbol
                        local_piano_tensor.append(self.symbol2index_piano[frame_piano])
                        local_piano_tensor.append(self.symbol2index_piano[STOP_SYMBOL])
                    else:
                        local_piano_tensor.append(self.symbol2index_piano[frame_piano])
                    orchestra_t_encoded = self.precomputed_vectors_orchestra[frame_orchestra].clone().detach()
                    orchestra_instruments_presence_t_encoded = self.precomputed_vectors_orchestra_instruments_presence[
                        PAD_SYMBOL].clone().detach()
                else:
                    local_piano_tensor = self.pianoroll_to_piano_tensor(
                        pr=this_pr_piano,
                        onsets=this_onsets_piano,
                        previous_frame_index=previous_frame_piano,
                        frame_index=frame_piano,
                        piano_vector=local_piano_tensor)

                    #  Time shift piano
                    local_piano_tensor.append(self.symbol2index_piano[TIME_SHIFT])

                    orchestra_t_encoded, previous_notes_orchestra, orchestra_instruments_presence_t_encoded = self.pianoroll_to_orchestral_tensor(
                        this_pr_orchestra,
                        this_onsets_orchestra,
                        previous_notes=previous_notes_orchestra,
                        frame_index=frame_orchestra)

                if orchestra_t_encoded is None:
                    avoid_this_chunk = True
                    break

                local_orchestra_tensor.append(orchestra_t_encoded)
                local_orchestra_instruments_presence_tensor.append(orchestra_instruments_presence_t_encoded)

            # If last symbol is not a PAD  Replace the last TIME_SHIFT added by a STOP
            if local_piano_tensor[-1] == self.symbol2index_piano[TIME_SHIFT]:
                local_piano_tensor.pop()
                local_piano_tensor.append(self.symbol2index_piano[STOP_SYMBOL])

            #  Pad piano vector to max length
            padding_length = self.max_number_messages_piano - len(local_piano_tensor)
            if padding_length < 0:
                raise Exception(
                    f"Padding length is equal to {padding_length}, consider increasing the max length of the encoding vector")
            self.smallest_padding_length = min(self.smallest_padding_length, padding_length)
            for _ in range(padding_length):
                local_piano_tensor.append(self.symbol2index_piano[PAD_SYMBOL])
            ############################################################

            if avoid_this_chunk:
                too_many_instruments_frame += 1
                continue

            assert len(local_piano_tensor) == self.max_number_messages_piano
            assert len(local_orchestra_tensor) == self.sequence_size

            local_piano_tensor = torch.LongTensor(local_piano_tensor)
            local_orchestra_tensor = torch.stack(local_orchestra_tensor)
            local_orchestra_instruments_presence_tensor = torch.stack(local_orchestra_instruments_presence_tensor)

            piano_tensor_dataset.append(
                local_piano_tensor[None, :].int())
            orchestra_tensor_dataset.append(
                local_orchestra_tensor[None, :, :].int())
            orchestra_instruments_presence_tensor_dataset.append(
                local_orchestra_instruments_presence_tensor[None, :, :].int())

        return minimum_transposition_allowed, maximum_transposition_allowed, \
               piano_tensor_dataset, orchestra_tensor_dataset, orchestra_instruments_presence_tensor_dataset, \
               total_chunk_counter, too_many_instruments_frame, impossible_transposition

    def pianoroll_to_piano_tensor(self, pr, onsets, previous_frame_index, frame_index, piano_vector):

        notes = [(e, True) for e in list(np.where(onsets[frame_index])[0])]
        if previous_frame_index is not None:
            delta_pr = pr[previous_frame_index] - pr[frame_index]
            notes += [(e, False) for e in list(np.where(delta_pr)[0])]

        # Sort notes from lowest to highest
        #  Todo check register individually for each voice for smaller categorical representations
        notes = sorted(notes, key=lambda e: e[0])

        #  Build symbols and append corresponding index in midi_messages
        previous_note = None
        for (note, on) in notes:
            #  Filter out note_off for repeated notes (uselessely crowds representation)
            if (previous_note == note) and (not on):
                continue
            if on:
                symbol = str(note) + '_on'
            else:
                symbol = str(note) + '_off'

            if symbol in self.symbol2index_piano.keys():
                index = self.symbol2index_piano[symbol]
            else:
                print(f'OOR: Piano - {symbol}')
                continue
            piano_vector.append(index)
            previous_note = note
        return piano_vector

    def init_generation_filepath(self, batch_size, context_length, filepath, banned_instruments=[],
                                 unknown_instruments=[], subdivision=None):

        ts_ind = self.symbol2index_piano[TIME_SHIFT]
        start_ind = self.symbol2index_piano[START_SYMBOL]
        pad_ind = self.symbol2index_piano[PAD_SYMBOL]
        end_ind = self.symbol2index_piano[END_SYMBOL]
        meta_inds = [ts_ind, start_ind, end_ind, pad_ind]
        stop_ind = self.symbol2index_piano[STOP_SYMBOL]

        # Get pianorolls
        score_piano = music21.converter.parse(filepath)

        if subdivision is None:
            subdivision = self.subdivision
        pianoroll_piano, onsets_piano, _ = score_to_pianoroll(
            score=score_piano,
            subdivision=self.subdivision,
            simplify_instrumentation=None,
            instrument_grouping=self.instrument_grouping,
            transpose_to_sounding_pitch=self.transpose_to_sounding_pitch,
            integrate_discretization=self.integrate_discretization,
            binarize=True
        )

        rhythm_piano = new_events(pianoroll_piano, onsets_piano)
        onsets_piano = onsets_piano["Piano"]
        piano_init_list = []
        previous_frame_index = None
        for frame_index in rhythm_piano:
            piano_init_list = self.pianoroll_to_piano_tensor(
                pr=pianoroll_piano["Piano"],
                onsets=onsets_piano,
                previous_frame_index=previous_frame_index,
                frame_index=frame_index,
                piano_vector=piano_init_list)
            previous_frame_index = frame_index

            #  Time shift piano
            piano_init_list.append(ts_ind)

        #  Remove the last time shift added
        piano_init_list.pop()

        # Prepend rests frames at the beginning and end of the piano score
        piano_init_list = [pad_ind] * (context_length - 1) + \
                          [start_ind] + \
                          piano_init_list + \
                          [end_ind] + \
                          [pad_ind] * (context_length - 1)

        # Use a non-sliced version for writing back the piano file
        piano_write = torch.tensor(piano_init_list).unsqueeze(0).repeat(batch_size, 1)

        #  Prepare chunks with padding for piano input
        def write_piano_chunk(piano_init, this_chunk, chunk_index, writing_stop_flag):
            piano_init[chunk_index, :len(this_chunk)] = torch.tensor(this_chunk)
            if writing_stop_flag:
                piano_init[chunk_index, len(this_chunk)] = stop_ind
                piano_init[chunk_index, len(this_chunk) + 1:] = pad_ind
            else:
                piano_init[chunk_index, len(this_chunk):] = pad_ind
            return piano_init

        piano_init = torch.zeros(len(rhythm_piano), self.max_number_messages_piano)
        this_chunk = []
        chunk_index = 0
        frame_counter = 0
        writing_stop_flag = True

        for piano_symbol in piano_init_list:
            if piano_symbol in meta_inds:
                if frame_counter >= self.sequence_size - 1:
                    #  Write in piano_init
                    piano_init = write_piano_chunk(piano_init, this_chunk, chunk_index, writing_stop_flag)
                    #  Clean this_chunk
                    while this_chunk[0] not in meta_inds:
                        this_chunk.pop(0)
                    this_chunk.pop(0)
                    chunk_index += 1
                frame_counter += 1

            if piano_symbol == end_ind:
                this_chunk.append(end_ind)
                this_chunk.append(stop_ind)
                writing_stop_flag = False
            else:
                this_chunk.append(piano_symbol)

        #  Don't forget last chunk
        piano_init = write_piano_chunk(piano_init, this_chunk, chunk_index, writing_stop_flag)

        # Orchestra
        padding_length = context_length * 2
        num_frames = len(rhythm_piano) + padding_length
        orchestra_silences, orchestra_unknown, instruments_presence, orchestra_init = \
            self.init_orchestra(num_frames, context_length, banned_instruments, unknown_instruments)

        # Repeat along batch dimension to generate several orchestation of the same piano score
        piano_init = piano_init.unsqueeze(0).repeat(batch_size, 1, 1)
        orchestra_init = orchestra_init.unsqueeze(0).repeat(batch_size, 1, 1)
        instruments_presence_init = instruments_presence.unsqueeze(0).repeat(batch_size, 1, 1)

        return piano_init.long(), piano_write.long(), rhythm_piano, \
               orchestra_init.long(), \
               instruments_presence_init.long(), orchestra_silences, orchestra_unknown

    def piano_tensor_to_score(self, tensor_score, durations=None, writing_tempo="adagio", subdivision=None):
        """

        :param durations:
        :param tensor_score: one-hot encoding with dimensions (time, instrument)
        :return:
        """
        # (batch, num_parts, notes_encoding)
        elems = list(tensor_score.numpy())

        if subdivision is None:
            subdivision = self.subdivision

        if durations is None:
            # Note: using len(elems) yields a duration vector which is too long, but fuck it
            durations = np.ones((len(elems))) * subdivision

        #  Init score
        stream = music21.stream.Stream()
        this_part = music21.stream.Part(id='Piano')
        music21_instrument = music21.instrument.fromString('Piano')
        this_part.insert(0, music21_instrument)

        # Keep track of the duration of notes on
        notes_on_duration = {}
        frame_index = 0
        global_offset = 0
        for elem in elems:
            #  DEBUG
            if elem == -1:
                # This is for debugging, to separate between batches
                f = music21.note.Rest()
                f.quarterLength = 1
                this_part.insert(global_offset / subdivision, f)
                global_offset += subdivision
                continue

            #  Get symbol corresponding to one-hot encoding
            symbol = self.index2symbol_piano[elem]

            if symbol in [START_SYMBOL, END_SYMBOL, PAD_SYMBOL]:
                global_offset += durations[frame_index]
                # frame_index += 1
            elif symbol in [TIME_SHIFT, STOP_SYMBOL]:
                # print(f'TS')
                # Increment duration in notes_on_duration
                global_offset += durations[frame_index]
                for pitch, _ in notes_on_duration.items():
                    notes_on_duration[pitch]['duration'] += durations[frame_index]
                #  increment frame counter
                frame_index += 1

                if symbol == STOP_SYMBOL:
                    # Write the last notes_on
                    for pitch, values in notes_on_duration.items():
                        duration = values['duration']
                        offset = values['offset']
                        f = music21.note.Note(int(pitch))
                        f.volume.velocity = 60.
                        f.quarterLength = duration / subdivision
                        this_part.insert((offset / subdivision), f)
                    break
            else:
                pitch, on = re.split('_', symbol)

                # print(f'{pitch} {on}')

                #  Repeat case
                if on == 'on':
                    if pitch in notes_on_duration.keys():
                        #  Write this pitch with its associated duration
                        duration = notes_on_duration[pitch]['duration']
                        offset = notes_on_duration[pitch]['offset']
                        f = music21.note.Note(int(pitch))
                        f.volume.velocity = 60.
                        f.quarterLength = duration / subdivision
                        this_part.insert((offset / subdivision), f)
                    #  Either it's a repeat or not, add in notes_on_duration
                    #  Write 0, as durations are all updated when a time shift event is met
                    else:
                        #  If it's not a note repeat need to instantiate the note in the dict
                        notes_on_duration[pitch] = {}
                    notes_on_duration[pitch]['duration'] = 0
                    notes_on_duration[pitch]['offset'] = global_offset
                # Note off
                elif on == 'off':
                    # Write this pitch with its associated duration
                    if pitch not in notes_on_duration.keys():
                        continue
                    duration = notes_on_duration[pitch]['duration']
                    offset = notes_on_duration[pitch]['offset']
                    f = music21.note.Note(int(pitch))
                    f.volume.velocity = 60.
                    f.quarterLength = duration / subdivision
                    this_part.insert((offset / subdivision), f)
                    #  Remove from notes_on_duration
                    notes_on_duration.pop(pitch, None)
                else:
                    raise Exception("should not happen")

        this_part.atSoundingPitch = self.transpose_to_sounding_pitch
        stream.append(this_part)

        return stream

    def tensor_to_score(self, tensor_score, score_type):
        if score_type == 'piano':
            return self.piano_tensor_to_score(tensor_score)
        elif score_type == 'orchestra':
            return self.orchestra_tensor_to_score(tensor_score)
        else:
            raise Exception(f"Expected score_type to be either piano or orchestra. Got {score_type} instead.")

    def visualise_batch(self, piano_pianoroll, orchestra_pianoroll, durations_piano, writing_dir,
                        filepath, writing_tempo, subdivision, only_orchestra):
        # data is a matrix (batch, ...)
        # Visualise a few examples
        if writing_dir is None:
            writing_dir = f"{self.dump_folder}/arrangement"

        if len(piano_pianoroll.size()) == 1:
            piano_pianoroll = torch.unsqueeze(piano_pianoroll, dim=0)
            orchestra_pianoroll = torch.unsqueeze(orchestra_pianoroll, dim=0)

        num_batches = len(piano_pianoroll)

        for batch_ind in range(num_batches):
            piano_part = self.piano_tensor_to_score(piano_pianoroll[batch_ind], durations_piano,
                                                    writing_tempo=writing_tempo,
                                                    subdivision=subdivision)
            orchestra_stream = self.orchestra_tensor_to_score(orchestra_pianoroll[batch_ind], durations_piano,
                                                              writing_tempo=writing_tempo,
                                                              subdivision=subdivision)

            orchestra_stream.write(fp=f"{writing_dir}/{filepath}_{batch_ind}_orchestra.mid", fmt='midi')
            if not only_orchestra:
                piano_part.write(fp=f"{writing_dir}/{filepath}_{batch_ind}_piano.mid", fmt='midi')
                # Both in the same score
                orchestra_stream.append(piano_part)
                orchestra_stream.write(fp=f"{writing_dir}/{filepath}_{batch_ind}_both.mid", fmt='midi')


if __name__ == '__main__':
    #  Read
    from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator

    dataset_manager_path = os.path.dirname(os.path.realpath(DatasetManager.__file__))
    database_path = f'{dataset_manager_path}/databases'
    dump_folder = f'{dataset_manager_path}/dump'

    # parameters
    sequence_size = 5
    max_transposition = 12
    velocity_quantization = 2
    subdivision = 16
    integrate_discretization = True

    transposition_semi_tone = 0

    corpus_it_gen = ArrangementIteratorGenerator(
        arrangement_path=f'{database_path}/Orchestration/arrangement',
        subsets=[
            # 'liszt_classical_archives',
            'debug',
        ],
        num_elements=None
    )

    dataset = ArrangementMidipianoDataset(corpus_it_gen,
                                          corpus_it_gen_instru_range=None,
                                          name="shit",
                                          mean_number_messages_per_time_frame=10,
                                          subdivision=subdivision,
                                          sequence_size=sequence_size,
                                          max_transposition=max_transposition,
                                          integrate_discretization=True,
                                          alignement_type='complete',
                                          transpose_to_sounding_pitch=True,
                                          compute_statistics_flag=None)

    dataset.load_index_dicts()

    writing_dir = f'{dump_folder}/arrangement_voice/reconstruction_midi'
    if os.path.isdir(writing_dir):
        shutil.rmtree(writing_dir)
    os.makedirs(writing_dir)

    num_frames = 500

    for arr_pair in dataset.iterator_gen():
        ######################################################################
        # Piano
        # Tensor
        arr_id = arr_pair['name']

        pr_piano, onsets_piano, _ = score_to_pianoroll(
            score=arr_pair['Piano'],
            subdivision=subdivision,
            simplify_instrumentation=None,
            instrument_grouping=dataset.instrument_grouping,
            transpose_to_sounding_pitch=dataset.transpose_to_sounding_pitch,
            integrate_discretization=dataset.integrate_discretization,
            binarize=True
        )

        events_piano = new_events(pr_piano, onsets_piano)
        events_piano = events_piano[:num_frames]
        piano_tensor = []
        previous_frame_index = None
        for frame_index in events_piano:
            piano_tensor = dataset.pianoroll_to_piano_tensor(
                pr=pr_piano['Piano'],
                onsets=onsets_piano['Piano'],
                previous_frame_index=previous_frame_index,
                frame_index=frame_index,
                piano_vector=piano_tensor)
            previous_frame_index = frame_index

            #  Time shift piano
            piano_tensor.append(dataset.symbol2index_piano[TIME_SHIFT])

        #  Remove the last time shift added
        piano_tensor.pop()

        piano_tensor = torch.LongTensor(piano_tensor)

        # Reconstruct
        piano_cpu = piano_tensor.cpu()
        duration_piano = list(np.asarray(events_piano)[1:] - np.asarray(events_piano)[:-1]) + [subdivision]

        piano_part = dataset.piano_tensor_to_score(piano_cpu, duration_piano, subdivision=subdivision)
        # piano_part.write(fp=f"{writing_dir}/{arr_id}_piano.xml", fmt='musicxml')
        piano_part.write(fp=f"{writing_dir}/{arr_id}_piano.mid", fmt='midi')

        ######################################################################
        #  Orchestra
        pr_orchestra, onsets_orchestra, _ = score_to_pianoroll(
            score=arr_pair['Orchestra'],
            subdivision=subdivision,
            simplify_instrumentation=dataset.simplify_instrumentation,
            instrument_grouping=dataset.instrument_grouping,
            transpose_to_sounding_pitch=dataset.transpose_to_sounding_pitch,
            integrate_discretization=dataset.integrate_discretization,
            binarize=True
        )

        events_orchestra = new_events(pr_orchestra, onsets_orchestra)
        events_orchestra = events_orchestra[:num_frames]
        orchestra_tensor = []
        previous_notes_orchestra = None
        for frame_counter, frame_index in enumerate(events_orchestra):
            orchestra_t_encoded, previous_notes_orchestra, _ = dataset.pianoroll_to_orchestral_tensor(
                pr=pr_orchestra,
                onsets=onsets_orchestra,
                previous_notes=previous_notes_orchestra,
                frame_index=frame_index)
            if orchestra_t_encoded is None:
                orchestra_t_encoded = dataset.precomputed_vectors_orchestra[REST_SYMBOL]
            orchestra_tensor.append(orchestra_t_encoded)

        orchestra_tensor = torch.stack(orchestra_tensor)

        # Reconstruct
        orchestra_cpu = orchestra_tensor.cpu()
        duration_orchestra = list(np.asarray(events_orchestra)[1:] - np.asarray(events_orchestra)[:-1]) + [
            subdivision]
        orchestra_part = dataset.orchestra_tensor_to_score(orchestra_cpu, duration_orchestra,
                                                           subdivision=subdivision)
        orchestra_part.write(fp=f"{writing_dir}/{arr_id}_orchestra.mid", fmt='midi')

        ######################################################################
        # Original
        try:
            arr_pair["Orchestra"].write(fp=f"{writing_dir}/{arr_id}_original.mid", fmt='midi')
            arr_pair["Piano"].write(fp=f"{writing_dir}/{arr_id}_original_piano.mid", fmt='midi')
        except:
            print("Can't write original")

        ######################################################################
        # Aligned version
        corresponding_frames = dataset.align_score(piano_pr=pr_piano,
                                                   piano_onsets=onsets_piano,
                                                   orchestra_pr=pr_orchestra,
                                                   orchestra_onsets=onsets_orchestra)

        corresponding_frames = corresponding_frames[:num_frames]

        piano_frames = [e[0][0] for e in corresponding_frames]
        orchestra_frames = [e[1][0] for e in corresponding_frames]

        piano_tensor_event = []
        orchestra_tensor_event = []
        previous_notes_orchestra = None
        for frame_counter, (frame_piano, frame_orchestra) in enumerate(zip(piano_frames, orchestra_frames)):

            #  IMPORTANT:
            #  Compute orchestra first to know if the frame has to be skipped or not
            #  (typically if too many instruments are played in one section)

            #######
            # Orchestra
            orchestra_t_encoded, previous_notes_orchestra, orchestra_instruments_presence_t_encoded = \
                dataset.pianoroll_to_orchestral_tensor(
                    pr=pr_orchestra,
                    onsets=onsets_orchestra,
                    previous_notes=previous_notes_orchestra,
                    frame_index=frame_orchestra
                )
            if orchestra_t_encoded is None:
                avoid_this_chunk = True
                continue
            orchestra_tensor_event.append(orchestra_t_encoded)

            #######
            # Piano
            piano_tensor_event = dataset.pianoroll_to_piano_tensor(
                pr=pr_piano['Piano'],
                onsets=onsets_piano['Piano'],
                previous_frame_index=previous_frame_index,
                frame_index=frame_piano,
                piano_vector=piano_tensor_event
            )
            previous_frame_index = frame_piano
            #  Time shift piano
            piano_tensor_event.append(dataset.symbol2index_piano[TIME_SHIFT])

        #  Remove the last time shift added
        piano_tensor_event.pop()

        piano_tensor_event = torch.LongTensor(piano_tensor_event)
        orchestra_tensor_event = torch.stack(orchestra_tensor_event)
        # Reconstruct
        orchestra_cpu = orchestra_tensor_event.cpu()
        orchestra_part = dataset.orchestra_tensor_to_score(orchestra_cpu, durations=None, subdivision=subdivision)
        piano_cpu = piano_tensor_event.cpu()
        piano_part = dataset.piano_tensor_to_score(piano_cpu, durations=None, subdivision=subdivision)
        orchestra_part.append(piano_part)
        orchestra_part.write(fp=f"{writing_dir}/{arr_id}_both_aligned.mid", fmt='midi')
