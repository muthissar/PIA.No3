"""
Compute prealigned version of the database
"""
import os
import shutil

import numpy as np
import torch
from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset

if __name__ == '__main__':
    #  Read
    from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator, score_to_pianoroll

    writing_dir = '/home/leo/Recherche/Code/DatasetManager/DatasetManager/dump/arrangement/aligned'
    subdivision = 32
    sequence_size = 5
    velocity_quantization = 128
    max_transposition = 0
    integrate_discretization = True

    database_path = f'{os.path.expanduser("~")}/Data/databases'
    corpus_it_gen = ArrangementIteratorGenerator(
        arrangement_path=f'{database_path}/Orchestration/arrangement',
        subsets=[
            'liszt_classical_archives',
            'bouliane',
            'hand_picked_Spotify',
            'kunstderfuge'
        ],
        num_elements=None
    )
    corpus_it_gen_instru_range = None

    dataset = ArrangementDataset(corpus_it_gen=corpus_it_gen,
                                 corpus_it_gen_instru_range=corpus_it_gen_instru_range,
                                 name='shit',
                                 subdivision=subdivision,
                                 sequence_size=sequence_size,
                                 velocity_quantization=velocity_quantization,
                                 max_transposition=max_transposition,
                                 integrate_discretization=integrate_discretization,
                                 alignement_type='complete',
                                 transpose_to_sounding_pitch=True,
                                 compute_statistics_flag=None)

    dataset.load_index_dicts()

    for arr_pair in dataset.iterator_gen():
        arr_id = arr_pair['name']
        ############################################################
        # Compute original pianorolls
        pianoroll_piano, onsets_piano, _ = score_to_pianoroll(
            score=arr_pair['Piano'],
            subdivision=dataset.subdivision,
            simplify_instrumentation=None,
            instrument_grouping=dataset.instrument_grouping,
            transpose_to_sounding_pitch=dataset.transpose_to_sounding_pitch,
            integrate_discretization=dataset.integrate_discretization,
            binarize=False,
        )

        pianoroll_orchestra, onsets_orchestra, _ = score_to_pianoroll(
            score=arr_pair['Orchestra'],
            subdivision=dataset.subdivision,
            simplify_instrumentation=dataset.simplify_instrumentation,
            instrument_grouping=dataset.instrument_grouping,
            transpose_to_sounding_pitch=dataset.transpose_to_sounding_pitch,
            integrate_discretization=dataset.integrate_discretization,
            binarize=False,
        )

        ############################################################
        #  Align (we can use non transposed scores, changes nothing to the alignement
        corresponding_frames = dataset.align_score(piano_pr=pianoroll_piano,
                                                   piano_onsets=onsets_piano,
                                                   orchestra_pr=pianoroll_orchestra,
                                                   orchestra_onsets=onsets_orchestra)
        piano_frames = [e[0][0] for e in corresponding_frames]
        orchestra_frames = [e[1][0] for e in corresponding_frames]

        piano_tensor_event = []
        orchestra_tensor_event = []
        previous_frame_index = None
        previous_notes_orchestra = None
        for frame_counter, (frame_piano, frame_orchestra) in enumerate(zip(piano_frames, orchestra_frames)):

            #  IMPORTANT:
            #  Compute orchestra first to know if the frame has to be skipped or not
            #  (typically if too many instruments are played in one section)

            #######
            # Orchestra
            orchestra_t_encoded, previous_notes_orchestra, orchestra_instruments_presence_t_encoded = \
                dataset.pianoroll_to_orchestral_tensor(
                    pr=pianoroll_orchestra,
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
            piano_t = dataset.pianoroll_to_piano_tensor(
                pr=pianoroll_piano['Piano'],
                onsets=onsets_piano['Piano'],
                frame_index=frame_piano)
            piano_tensor_event.append(piano_t)

        piano_tensor_event = torch.stack(piano_tensor_event)
        orchestra_tensor_event = torch.stack(orchestra_tensor_event)
        # Reconstruct
        duration_piano = list(np.asarray(piano_frames)[1:] - np.asarray(piano_frames)[:-1]) + [subdivision]
        orchestra_cpu = orchestra_tensor_event.cpu()
        orchestra_stream, _, _ = dataset.orchestra_tensor_to_score(orchestra_cpu,
                                                                   durations=duration_piano,
                                                                   subdivision=subdivision,
                                                                   format='mid')
        piano_cpu = piano_tensor_event.cpu()
        piano_part = dataset.piano_tensor_to_score(piano_cpu,
                                                   durations=duration_piano,
                                                   subdivision=subdivision,
                                                   format='mid')

        this_folder = f"{writing_dir}/{arr_id}"
        if not os.path.isdir(this_folder):
            os.mkdir(this_folder)
        orchestra_stream.write(fp=f"{this_folder}/orchestra.mid", fmt='midi')
        piano_part.write(fp=f"{this_folder}/piano.mid", fmt='midi')
