import os
import pickle
import shutil

import music21
import numpy as np
import torch

import DatasetManager
from DatasetManager.arrangement.arrangement_helper import score_to_pianoroll, shift_pr_along_pitch_axis, \
    note_to_midiPitch, new_events
from DatasetManager.helpers import REST_SYMBOL, SLUR_SYMBOL, END_SYMBOL, START_SYMBOL, \
    YES_SYMBOL, NO_SYMBOL, PAD_SYMBOL
from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset
from tqdm import tqdm

"""
Piano is encoded like orchestra.
It has a fixed number of voices V, and each temporal frame is a vector
[v_1, ..., v_V-1]
where v_i is a categorical variable whose labels are {p_i, SLUR, START, END, REST, PAD}, 
where p_i represent pitch and s_i represents a slured pitch.
We use a different slur symbol for each pitch to avoid ambiguous situations where the position in the voices are shifted,
like for instance if the lowest note at time t is slured, but at time t+1 a new note starts plaing at a lower pitch.
With a single slur symbol, encoding would be:
@t : [p_low, X, ..., X]
@t+1 : [p_lower, slur, X, ..., X]
and the slur seems to be on the X symbol

To prevent this, slured notes are writen at the same voice, onsets note fills free slots

Hence a chunk is represented as the concatentation of frames 
[v_0^0, ..., v_(V-1)^0, v_0^1, ..., v_(V-1)^1, ...]

Voices are ordered from lowest to highest.
"""


class ArrangementVoiceDataset(ArrangementDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 corpus_it_gen_instru_range,
                 name,
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

        # Mapping between instruments and indices
        self.index2instrument = {}
        self.instrument2index = {}
        self.index2midi_pitch = {}
        self.midi_pitch2index = {}
        # Instruments presence
        self.instruments_presence2index = {}
        self.index2instruments_presence = {}
        self.instrument_presence_name2index = {}
        self.instrument_presence_index2name = {}
        #  Piano
        self.index2midi_pitch_piano = {}
        self.midi_pitch_piano2index = {}
        # Dimensions
        self.number_instruments = None
        self.number_voices_piano = self.instrumentation["Piano"]
        self.instrument_presence_dim = None

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
        name = f'ArrangementVoiceDataset-' \
            f'{self.name}-' \
            f'{self.subdivision}-' \
            f'{self.sequence_size}-' \
            f'{self.max_transposition}'
        return name

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
        # Instruments presence
        self.instruments_presence2index = index_dicts['instruments_presence2index']
        self.index2instruments_presence = index_dicts['index2instruments_presence']
        self.instrument_presence_name2index = index_dicts['instrument_presence_name2index']
        self.instrument_presence_index2name = index_dicts['instrument_presence_index2name']
        #  Piano
        self.index2midi_pitch_piano = index_dicts['index2midi_pitch_piano']
        self.midi_pitch_piano2index = index_dicts['midi_pitch_piano2index']
        # Dimensions
        self.number_instruments = index_dicts['number_instruments']
        self.instrument_presence_dim = index_dicts['instrument_presence_dim']
        # Misc
        self.observed_tessitura = index_dicts['observed_tessitura']

        ############################################################
        ############################################################
        # These are the one-hot representation of several useful (especially during generation) vectors
        piano_rest_vector = [self.midi_pitch_piano2index[REST_SYMBOL]] * self.number_voices_piano
        piano_start_vector = [self.midi_pitch_piano2index[START_SYMBOL]] * self.number_voices_piano
        piano_end_vector = [self.midi_pitch_piano2index[END_SYMBOL]] * self.number_voices_piano
        piano_pad_vector = [self.midi_pitch_piano2index[PAD_SYMBOL]] * self.number_voices_piano
        self.precomputed_vectors_piano[REST_SYMBOL] = torch.from_numpy(np.asarray(piano_rest_vector)).long()
        self.precomputed_vectors_piano[START_SYMBOL] = torch.from_numpy(np.asarray(piano_start_vector)).long()
        self.precomputed_vectors_piano[END_SYMBOL] = torch.from_numpy(np.asarray(piano_end_vector)).long()
        self.precomputed_vectors_piano[PAD_SYMBOL] = torch.from_numpy(np.asarray(piano_pad_vector)).long()

        orchestra_start_vector = []
        orchestra_end_vector = []
        orchestra_rest_vector = []
        orchestra_pad_vector = []
        orchestra_mask_vector = []
        for instru_ind, mapping in self.midi_pitch2index.items():
            orchestra_start_vector.append(mapping[START_SYMBOL])
            orchestra_end_vector.append(mapping[END_SYMBOL])
            orchestra_rest_vector.append(mapping[REST_SYMBOL])
            orchestra_pad_vector.append(mapping[PAD_SYMBOL])
        self.precomputed_vectors_orchestra[START_SYMBOL] = torch.from_numpy(np.asarray(orchestra_start_vector)).long()
        self.precomputed_vectors_orchestra[END_SYMBOL] = torch.from_numpy(np.asarray(orchestra_end_vector)).long()
        self.precomputed_vectors_orchestra[REST_SYMBOL] = torch.from_numpy(np.asarray(orchestra_rest_vector)).long()
        self.precomputed_vectors_orchestra[PAD_SYMBOL] = torch.from_numpy(np.asarray(orchestra_pad_vector)).long()
        #
        unknown_vector = np.ones((self.instrument_presence_dim)) * self.instruments_presence2index[PAD_SYMBOL]
        self.precomputed_vectors_orchestra_instruments_presence[PAD_SYMBOL] = \
            torch.from_numpy(unknown_vector).long()
        ############################################################
        ############################################################
        return

    def compute_index_dicts(self, index_dict_path):

        if self.compute_tessitura:
            ############################################################
            ############################################################
            #  Mapping midi_pitch to token for each instrument
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

                pianoroll_orchestra, onsets_orchestra, _ = score_to_pianoroll(
                    score=arr_pair['Orchestra'],
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

                    pianoroll_orchestra, _, _ = score_to_pianoroll(arr_pair['Orchestra'], self.subdivision,
                                                                   self.simplify_instrumentation,
                                                                   self.instrument_grouping,
                                                                   self.transpose_to_sounding_pitch,
                                                                   self.integrate_discretization,
                                                                   binarize=True)
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
        # Save this in a file
        if self.compute_statistics_flag:
            with open(f"{self.statistic_folder}/note_frequency_per_instrument", "w") as ff:
                for instrument_name, set_pitch_class in set_midiPitch_per_instrument.items():
                    ff.write(f"# {instrument_name}: \n")
                    for pc in set_pitch_class:
                        ff.write(f"   {pc}\n")

        ############################################################
        # Local dicts used temporarily
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
            index = 0
            for midi_pitch in list_midiPitch:
                midi_pitch2index_per_instrument[instrument_name][midi_pitch] = index
                index2midi_pitch_per_instrument[instrument_name][index] = midi_pitch
                index += 1
            # for midi_pitch in list_midiPitch:
            #     midi_pitch2index_per_instrument[instrument_name][f's_{midi_pitch}'] = index
            #     index2midi_pitch_per_instrument[instrument_name][index] = f's_{midi_pitch}'
            #     index += 1
            # Silence
            midi_pitch2index_per_instrument[instrument_name][REST_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = REST_SYMBOL
            index += 1
            #  Slur
            midi_pitch2index_per_instrument[instrument_name][SLUR_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = SLUR_SYMBOL
            index += 1
            #  Pad
            midi_pitch2index_per_instrument[instrument_name][PAD_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = PAD_SYMBOL
            index += 1
            # Start
            midi_pitch2index_per_instrument[instrument_name][START_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = START_SYMBOL
            index += 1
            # End
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
        # Piano
        lower_note_piano, higher_note_piano = self.reference_tessitura["Piano"]
        min_pitch_piano = note_to_midiPitch(lower_note_piano)
        max_pitch_piano = note_to_midiPitch(higher_note_piano)
        self.index2midi_pitch_piano = {}
        self.midi_pitch_piano2index = {}
        index = 0
        for pitch_name in range(min_pitch_piano, max_pitch_piano):
            self.index2midi_pitch_piano[index] = pitch_name
            self.midi_pitch_piano2index[pitch_name] = index
            index += 1
        # for pitch_name in range(min_pitch_piano, max_pitch_piano):
        #     self.index2midi_pitch_piano[index] = f's_{pitch_name}'
        #     self.midi_pitch_piano2index[f's_{pitch_name}'] = index
        #     index += 1
        # Silence
        self.midi_pitch_piano2index[REST_SYMBOL] = index
        self.index2midi_pitch_piano[index] = REST_SYMBOL
        index += 1
        #  Slur
        self.midi_pitch_piano2index[SLUR_SYMBOL] = index
        self.index2midi_pitch_piano[index] = SLUR_SYMBOL
        index += 1
        # Pad
        self.midi_pitch_piano2index[PAD_SYMBOL] = index
        self.index2midi_pitch_piano[index] = PAD_SYMBOL
        index += 1
        # Start
        self.midi_pitch_piano2index[START_SYMBOL] = index
        self.index2midi_pitch_piano[index] = START_SYMBOL
        index += 1
        # End
        self.midi_pitch_piano2index[END_SYMBOL] = index
        self.index2midi_pitch_piano[index] = END_SYMBOL
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
            'instruments_presence2index': self.instruments_presence2index,
            'index2instruments_presence': self.index2instruments_presence,
            'instrument_presence_name2index': self.instrument_presence_name2index,
            'instrument_presence_index2name': self.instrument_presence_index2name,
            'index2midi_pitch_piano': self.index2midi_pitch_piano,
            'midi_pitch_piano2index': self.midi_pitch_piano2index,
            'number_instruments': self.number_instruments,
            'instrument_presence_dim': self.instrument_presence_dim,
            'observed_tessitura': self.observed_tessitura
        }

        with open(index_dict_path, 'wb') as ff:
            pickle.dump(index_dicts, ff)

        return

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
            previous_notes_piano = None
            for frame_piano, frame_orchestra in zip(this_chunk_piano_indices, this_chunk_orchestra_indices):
                # Piano encoded vector
                if frame_orchestra in [START_SYMBOL, END_SYMBOL, PAD_SYMBOL]:
                    assert frame_piano in [START_SYMBOL, END_SYMBOL, PAD_SYMBOL], 'problem'
                    #  Padding vectors at beginning or end
                    piano_t_encoded = self.precomputed_vectors_piano[frame_piano].clone().detach()
                    orchestra_t_encoded = self.precomputed_vectors_orchestra[frame_orchestra].clone().detach()
                    orchestra_instruments_presence_t_encoded = \
                        self.precomputed_vectors_orchestra_instruments_presence[PAD_SYMBOL].clone().detach()
                else:
                    piano_t_encoded, previous_notes_piano = self.pianoroll_to_piano_tensor(
                        pr=this_pr_piano,
                        onsets=this_onsets_piano,
                        previous_notes=previous_notes_piano,
                        frame_index=frame_piano)

                    orchestra_t_encoded, previous_notes_orchestra, orchestra_instruments_presence_t_encoded = \
                        self.pianoroll_to_orchestral_tensor(
                            pr=this_pr_orchestra,
                            onsets=this_onsets_orchestra,
                            previous_notes=previous_notes_orchestra,
                            frame_index=frame_orchestra
                        )

                if orchestra_t_encoded is None:
                    avoid_this_chunk = True
                    break

                if piano_t_encoded is None:
                    avoid_this_chunk = True
                    break

                local_piano_tensor.append(piano_t_encoded)
                local_orchestra_tensor.append(orchestra_t_encoded)
                local_orchestra_instruments_presence_tensor.append(orchestra_instruments_presence_t_encoded)
            ############################################################

            if avoid_this_chunk:
                too_many_instruments_frame += 1
                continue

            assert len(local_piano_tensor) == self.sequence_size
            assert len(local_orchestra_tensor) == self.sequence_size

            local_piano_tensor = torch.stack(local_piano_tensor)
            local_orchestra_tensor = torch.stack(local_orchestra_tensor)
            local_orchestra_instruments_presence_tensor = torch.stack(local_orchestra_instruments_presence_tensor)

            piano_tensor_dataset.append(
                local_piano_tensor[None, :, :].int())
            orchestra_tensor_dataset.append(
                local_orchestra_tensor[None, :, :].int())
            orchestra_instruments_presence_tensor_dataset.append(
                local_orchestra_instruments_presence_tensor[None, :, :].int())

        return minimum_transposition_allowed, maximum_transposition_allowed, \
               piano_tensor_dataset, orchestra_tensor_dataset, orchestra_instruments_presence_tensor_dataset, \
               total_chunk_counter, too_many_instruments_frame, impossible_transposition

    def pianoroll_to_piano_tensor(self, pr, onsets, previous_notes, frame_index):

        piano_encoded = np.zeros((self.number_voices_piano)) - 1
        current_notes = {}

        # Get list of notes at frame_index
        if previous_notes is None:
            notes_onsets = [e for e in list(np.where(pr[frame_index])[0])]
            notes_slurs = []
        else:
            notes_onsets = [e for e in list(np.where(onsets[frame_index])[0])]
            notes_slurs = [e for e in list(np.where(pr[frame_index])[0]) if e not in notes_onsets]

        # Sort notes from lowest to highest
        notes_onsets = sorted(notes_onsets)
        notes_slurs = sorted(notes_slurs)

        #  First write Slurs at same location than slured not
        for note in notes_slurs:
            #  OOR ?
            if note not in self.midi_pitch_piano2index.keys():
                continue

            writen = False
            encoded_note = self.midi_pitch_piano2index[note]
            for previous_note, previous_index in previous_notes.items():
                if previous_note == encoded_note:
                    piano_encoded[previous_index] = self.midi_pitch_piano2index[SLUR_SYMBOL]
                    current_notes[encoded_note] = previous_index
                    writen = True
                    break
            if not writen:
                #  Can happen when:
                #  - its the first frame
                # - onset is not up to date anymore after automatic alignement (due to skipped frames)
                for index in range(self.number_voices_piano):
                    if piano_encoded[index] == -1:
                        encoded_note = self.midi_pitch_piano2index[note]
                        piano_encoded[index] = encoded_note
                        current_notes[encoded_note] = index
                        writen = True
                        break
            if not writen:
                raise Exception('Slur not writen')

        #  Write onsets notes at other locations
        for note in notes_onsets:
            # OOR ?
            if note not in self.midi_pitch_piano2index.keys():
                print(f'OOR: Piano - {note}')
                continue

            writen = False
            #  Find first free slot
            for index in range(self.number_voices_piano):
                if piano_encoded[index] == -1:
                    encoded_note = self.midi_pitch_piano2index[note]
                    piano_encoded[index] = encoded_note
                    current_notes[encoded_note] = index
                    writen = True
                    break
            if not writen:
                raise Exception('Piano vector is too small to encode this frame')

        location_to_rest = np.where(piano_encoded == -1)[0]
        piano_encoded[location_to_rest] = self.midi_pitch_piano2index[REST_SYMBOL]
        piano_vector = torch.tensor(piano_encoded).long()
        return piano_vector, current_notes

    def piano_tensor_to_score(self, tensor_score, durations=None, writing_tempo="adagio", subdivision=None):
        """

        :param durations:
        :param tensor_score: one-hot encoding with dimensions (time, instrument)
        :return:
        """
        # (batch, num_parts, notes_encoding)
        piano_matrix = tensor_score.numpy()
        length = len(piano_matrix)

        if subdivision is None:
            subdivision = self.subdivision

        if durations is None:
            durations = np.ones((length)) * subdivision
        else:
            assert length == len(durations), "Rhythm vector must be the same length as tensor[0]"

        # First store every in a dict {instrus : [time [notes]]}
        score_list = []
        for voice_index in range(self.number_voices_piano):
            offset = 0
            #  Each voice is monophonic
            for frame_index, duration in enumerate(durations):
                symbol = self.index2midi_pitch_piano[piano_matrix[frame_index, voice_index]]
                if symbol not in [START_SYMBOL, END_SYMBOL, REST_SYMBOL, PAD_SYMBOL]:
                    if symbol == SLUR_SYMBOL:
                        (this_pitch, this_offset, this_duration) = score_list.pop(-1)
                        new_elem = (this_pitch, this_offset, this_duration + duration)
                        score_list.append(new_elem)
                    else:
                        new_elem = (symbol, offset, duration)
                        score_list.append(new_elem)
                offset += duration

        #  Batch is used as time in the score
        stream = music21.stream.Stream()
        this_part = music21.stream.Part(id='Piano')
        music21_instrument = music21.instrument.fromString('Piano')
        this_part.insert(0, music21_instrument)
        #  Sort by offset time (not sure it's very useful, more for debugging purposes)
        score_list = sorted(score_list, key=lambda e: e[1])
        for elem in score_list:
            pitch, offset, duration = elem
            f = music21.note.Note(pitch)
            f.volume.velocity = 60.
            f.quarterLength = duration / subdivision
            this_part.insert((offset / subdivision), f)

        this_part.atSoundingPitch = self.transpose_to_sounding_pitch
        stream.append(this_part)

        return stream

    def visualise_batch(self, piano_pianoroll, orchestra_pianoroll, durations_piano, writing_dir,
                        filepath, writing_tempo, subdivision, only_orchestra):
        # data is a matrix (batch, ...)
        # Visualise a few examples
        if writing_dir is None:
            writing_dir = f"{self.dump_folder}/arrangement_voice"

        if len(piano_pianoroll.size()) == 2:
            piano_flat = piano_pianoroll
            orchestra_flat = orchestra_pianoroll
        else:
            # Add padding vectors between each example
            batch_size, time_length, num_features = piano_pianoroll.size()
            piano_with_padding_between_batch = torch.zeros(batch_size, time_length + 1, num_features)
            piano_with_padding_between_batch[:, :time_length] = piano_pianoroll
            piano_with_padding_between_batch[:, time_length] = self.precomputed_vectors_piano[REST_SYMBOL]
            piano_flat = piano_with_padding_between_batch.view(-1, self.number_voices_piano)
            #
            batch_size, time_length, num_features = orchestra_pianoroll.size()
            orchestra_with_padding_between_batch = torch.zeros(batch_size, time_length + 1, num_features)
            orchestra_with_padding_between_batch[:, :time_length] = orchestra_pianoroll
            orchestra_with_padding_between_batch[:, time_length] = self.precomputed_vectors_orchestra[REST_SYMBOL]
            orchestra_flat = orchestra_with_padding_between_batch.view(-1, self.number_instruments)

        piano_part = self.piano_tensor_to_score(piano_flat, durations_piano, writing_tempo=writing_tempo,
                                                subdivision=subdivision)
        orchestra_stream = self.orchestra_tensor_to_score(orchestra_flat, durations_piano, writing_tempo=writing_tempo,
                                                          subdivision=subdivision)

        orchestra_stream.write(fp=f"{writing_dir}/{filepath}_orchestra.mid", fmt='midi')
        if not only_orchestra:
            piano_part.write(fp=f"{writing_dir}/{filepath}_piano.mid", fmt='midi')
            # Both in the same score
            orchestra_stream.append(piano_part)
            orchestra_stream.write(fp=f"{writing_dir}/{filepath}_both.mid", fmt='midi')

    def init_generation_filepath(self, batch_size, context_length, filepath, banned_instruments=[],
                                 unknown_instruments=[], subdivision=None):
        # Get pianorolls
        score_piano = music21.converter.parse(filepath)

        if subdivision is None:
            subdivision = self.subdivision
        pianoroll_piano, onsets_piano, _ = score_to_pianoroll(score_piano,
                                                              subdivision,
                                                              simplify_instrumentation=None,
                                                              instrument_grouping=self.instrument_grouping,
                                                              transpose_to_sounding_pitch=self.transpose_to_sounding_pitch,
                                                              integrate_discretization=self.integrate_discretization,
                                                              binarize=True)

        rhythm_piano = new_events(pianoroll_piano, onsets_piano)
        onsets_piano = onsets_piano["Piano"]
        pr_piano = pianoroll_piano['Piano']
        piano_tensor = []
        previous_notes_piano = None
        for frame_index in rhythm_piano:
            piano_t_encoded, previous_notes_piano = self.pianoroll_to_piano_tensor(
                pr=pr_piano,
                onsets=onsets_piano,
                previous_notes=previous_notes_piano,
                frame_index=frame_index)
            piano_tensor.append(piano_t_encoded)

        # Prepend rests frames at the beginning and end of the piano score
        piano_tensor = [self.precomputed_vectors_piano[PAD_SYMBOL]] * (context_length - 1) + \
                       [self.precomputed_vectors_piano[START_SYMBOL]] + \
                       piano_tensor + \
                       [self.precomputed_vectors_piano[END_SYMBOL]] + \
                       [self.precomputed_vectors_piano[PAD_SYMBOL]] * (context_length - 1)

        piano_init = torch.stack(piano_tensor)

        # Orchestra
        num_frames = piano_init.shape[0]  #  Here batch size is time dimensions (each batch index is a piano event)
        orchestra_silences, orchestra_unknown, instruments_presence, orchestra_init = \
            self.init_orchestra(num_frames, context_length, banned_instruments, unknown_instruments)

        # Repeat along batch dimension to generate several orchestation of the same piano score
        piano_init = piano_init.unsqueeze(0).repeat(batch_size, 1, 1)
        orchestra_init = orchestra_init.unsqueeze(0).repeat(batch_size, 1, 1)
        instruments_presence_init = instruments_presence.unsqueeze(0).repeat(batch_size, 1, 1)
        piano_write = piano_init

        return piano_init.long(), piano_write.long(), rhythm_piano, \
               orchestra_init.long(), \
               instruments_presence_init.long(), orchestra_silences, orchestra_unknown


if __name__ == '__main__':
    #  Read
    from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator

    dataset_manager_path = os.path.dirname(os.path.realpath(DatasetManager.__file__))
    database_path = f'{dataset_manager_path}/databases'
    dump_folder = f'{dataset_manager_path}/dump'

    # parameters
    sequence_size = 5
    max_transposition = 3
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

    dataset = ArrangementVoiceDataset(corpus_it_gen=corpus_it_gen,
                                      corpus_it_gen_instru_range=None,
                                      name="shit",
                                      subdivision=subdivision,
                                      sequence_size=sequence_size,
                                      max_transposition=max_transposition,
                                      integrate_discretization=integrate_discretization,
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

        pr_piano, onsets_piano, _ = score_to_pianoroll(arr_pair['Piano'],
                                                       subdivision,
                                                       simplify_instrumentation=None,
                                                       instrument_grouping=dataset.instrument_grouping,
                                                       transpose_to_sounding_pitch=dataset.transpose_to_sounding_pitch,
                                                       integrate_discretization=dataset.integrate_discretization,
                                                       binarize=True)

        events_piano = new_events(pr_piano, onsets_piano)
        events_piano = events_piano[:num_frames]
        piano_tensor = []
        previous_notes_piano = None
        for frame_index in events_piano:
            piano_frame, previous_notes_piano = dataset.pianoroll_to_piano_tensor(
                pr=pr_piano['Piano'],
                onsets=onsets_piano['Piano'],
                previous_notes=previous_notes_piano,
                frame_index=frame_index)

            piano_tensor.append(piano_frame)

        piano_tensor = torch.stack(piano_tensor)

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
        instruments_presence_tensor = []
        previous_notes_orchestra = None
        for frame_counter, frame_index in enumerate(events_orchestra):
            orchestra_t_encoded, previous_notes_orchestra, orchestra_instruments_presence_t_encoded = dataset.pianoroll_to_orchestral_tensor(
                pr=pr_orchestra,
                onsets=onsets_orchestra,
                previous_notes=previous_notes_orchestra,
                frame_index=frame_index)
            if orchestra_t_encoded is None:
                orchestra_t_encoded = dataset.precomputed_vectors_orchestra[REST_SYMBOL]
            orchestra_tensor.append(orchestra_t_encoded)
            instruments_presence_tensor.append(orchestra_instruments_presence_t_encoded)

        orchestra_tensor = torch.stack(orchestra_tensor)
        instruments_presence_tensor = torch.stack(instruments_presence_tensor)

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

        ###################################
        # Transpose pianorolls
        if transposition_semi_tone != 0:
            pr_piano_transp = shift_pr_along_pitch_axis(pr_piano['Piano'], transposition_semi_tone)
            onsets_piano_transp = shift_pr_along_pitch_axis(onsets_piano['Piano'], transposition_semi_tone)

            pr_orchestra_transp = {}
            onsets_orchestra_transp = {}
            for instrument_name in pr_orchestra.keys():
                # Pr
                pr = pr_orchestra[instrument_name]
                shifted_pr = shift_pr_along_pitch_axis(pr, transposition_semi_tone)
                pr_orchestra_transp[instrument_name] = shifted_pr
                # Onsets
                onsets = onsets_orchestra[instrument_name]
                shifted_onsets = shift_pr_along_pitch_axis(onsets, transposition_semi_tone)
                onsets_orchestra_transp[instrument_name] = shifted_onsets
        else:
            pr_piano_transp = pr_piano['Piano']
            onsets_piano_transp = onsets_piano['Piano']
            pr_orchestra_transp = pr_orchestra
            onsets_orchestra_transp = onsets_orchestra
        ###################################

        piano_tensor_event = []
        orchestra_tensor_event = []
        previous_notes_piano = None
        previous_notes_orchestra = None
        for frame_counter, (frame_piano, frame_orchestra) in enumerate(zip(piano_frames, orchestra_frames)):

            #  IMPORTANT:
            #  Compute orchestra first to know if the frame has to be skipped or not
            #  (typically if too many instruments are played in one section)

            #######
            # Orchestra
            orchestra_t_encoded, previous_notes_orchestra, orchestra_instruments_presence_t_encoded = \
                dataset.pianoroll_to_orchestral_tensor(
                    pr=pr_orchestra_transp,
                    onsets=onsets_orchestra_transp,
                    previous_notes=previous_notes_orchestra,
                    frame_index=frame_orchestra
                )
            if orchestra_t_encoded is None:
                avoid_this_chunk = True
                continue
            orchestra_tensor_event.append(orchestra_t_encoded)

            #######
            # Piano
            piano_t, previous_notes_piano = dataset.pianoroll_to_piano_tensor(
                pr=pr_piano_transp,
                onsets=onsets_piano_transp,
                previous_notes=previous_notes_piano,
                frame_index=frame_piano)
            piano_tensor_event.append(piano_t)

        piano_tensor_event = torch.stack(piano_tensor_event)
        orchestra_tensor_event = torch.stack(orchestra_tensor_event)
        # Reconstruct
        orchestra_cpu = orchestra_tensor_event.cpu()
        orchestra_part = dataset.orchestra_tensor_to_score(orchestra_cpu, durations=None, subdivision=subdivision)
        piano_cpu = piano_tensor_event.cpu()
        piano_part = dataset.piano_tensor_to_score(piano_cpu, durations=None, subdivision=subdivision)
        orchestra_part.append(piano_part)
        orchestra_part.write(fp=f"{writing_dir}/{arr_id}_both_aligned.mid", fmt='midi')
