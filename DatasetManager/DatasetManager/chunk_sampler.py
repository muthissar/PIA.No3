import music21

from music21 import stream, meter

from DatasetManager.chorale_dataset import ChoraleDataset


class ChunkSampler(ChoraleDataset):
    """
    This class allows to sample stream chunks of chorales obtained by running
    a sliding window with chosen window size (in number of measures) over
    the chorales.

    In order to have a coherent meter, we only consider chorales with a 4/4
    time-signature.
    """
    def __init__(self,
                 chorale_dataset,
                 corpus_it_gen,
                 name,
                 voice_ids,
                 metadatas=None,
                 sequences_size_measures=4,
                 subdivision=4,
                 cache_dir=None):
        self.sequences_size_measures = sequences_size_measures
        sequence_size_quarters = sequences_size_measures*4
        super(ChunkSampler, self).__init__(
            corpus_it_gen=corpus_it_gen,
            name=name,
            voice_ids=voice_ids,
            metadatas=metadatas,
            sequences_size=sequence_size_quarters,
            subdivision=subdivision,
            cache_dir=cache_dir)
        self.windows = self._build_sliding_window()
        self.index2note_dicts = chorale_dataset.index2note_dicts
        self.note2index_dicts = chorale_dataset.note2index_dicts

    def get_random_chunk(self):
        raise NotImplementedError

    def is_valid(self, chorale: music21.stream.Stream):
        """
        Check validity of a choraleself.
        This uses the main validity check from the ChoraleDataset
        """
        main_is_valid = super(ChunkSampler, self).is_valid(chorale)
        if not main_is_valid:
            return False

        # keep only chorales in constant 4/4 time-signature
        time_signatures = chorale.flat.getTimeSignatures(returnDefault=False)
        has_constant_time_signature = len(time_signatures) == 4
        if not has_constant_time_signature:
            return False
        is4_by_4 = time_signatures[0].ratioEqual(meter.TimeSignature('4/4'))
        if not is4_by_4:
            return False

        fourQuarterNotesDuration = m21.duration.Duration(4.)
        allDuration4Quarters = all([
            measure.duration == fourQuarterNotesDuration
            for measure in chorale.semiFlat.getElementsByClass('Measure')])
        if not allDuration4Quarters:
            return False

        return True

    def sliding_window(self, chorale: music21.stream.Stream):
        sequence_size_measures = self.sequences_size / 4
        numSlices = chorale_duration_quarters // self.sequences_size

        windows = [
            {'chunk': chorale.measures(
                k*sequence_size_measures,
                (k+1)*sequence_size_measures - 1),
             'offset_start_quarters': k*sequence_size_measures * 4,
             'chorale': chorale.metadata.all,
             'metadata': TODO ADD METADATA HERE}
            for k in range(0, numSlices)]
        return windows

    def get_windows():
        windows = []
        for chorale in self.iterator_gen():
            windows.extend(self.sliding_window(chorale))
