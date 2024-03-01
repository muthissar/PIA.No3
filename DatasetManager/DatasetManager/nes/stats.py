from math import ceil, sqrt
import matplotlib.pyplot as plt

class Stats(dict):
    def __init__(self, iterable=None, **kwds):
        super(Stats, self).__init__()

        quantities = ['sum', 'squared_sum', 'total']
        self._cache = {q: None for q in quantities}
        self._cache['keys'] = []
        self._cache['values'] = []
        self._up_to_date = []

        self.update(iterable)


    # decorators
    class Decorators:
        @staticmethod
        def cached(method):
            name = method.__name__
            def wrapped(self, *args, **kwargs):
                if self._up_to_date.get(name, False):
                    return self._cache[name] # lambda: self._cache[name] ?
                result = method(self, *args, **kwargs)
                self._cache[name] = result
                self._up_to_date[name] = True
                return result
            return wrapped

        @staticmethod
        def setter(method):
            def wrapped(self, *args, **kwargs):
                self._up_to_date = {}
                return method(self, *args, **kwargs)
            return wrapped

        @staticmethod
        def plot(method):
            def wrapped(self, *args, show=False, title='', **kwargs):
                method(self, *args, **kwargs)
                plt.title(title)
                if show:
                    plt.show()
            return wrapped


    # private methods
    def __missing__(self, key):
        float(key) # avoid non-numeric keys
        self._cache['keys'].append(key)
        self._up_to_date['keys'] = False
        return 0

    def __repr__(self):
        return f'Stats({super(Stats, self).__repr__()})'

    def __str__(self):
        methods = [self.min, self.max, self.mean, self.std, self.q1, self.median, self.q3]
        stats = [f'{method.__name__}: {round(method(), 3)}' for method in methods]
        return '\t'.join(stats)

    # add data
    @Decorators.setter
    def update(self, iterable=None, **kwds):
        if iterable is not None:
            for elem in iterable:
                self[elem] += 1
        if kwds:
            self.update(kwds)

    # statistics methods
    @Decorators.cached
    def keys(self):
        return sorted(self._cache['keys'])

    @Decorators.cached
    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return zip(self.keys(), self.values())


    @Decorators.cached
    def sum(self):
        return sum(k * v for k, v in self.items())

    @Decorators.cached
    def squared_sum(self):
        return sum(k * k * v for k, v in self.items())

    @Decorators.cached
    def total(self):
        return sum(self.values())

    def min(self):
        return min(self.keys())

    def max(self):
        return max(self.keys())

    def mean(self):
        return self.sum() / self.total()

    def std(self):
        return sqrt(self.squared_sum() / self.total() - self.mean()**2)

    def quantile(self, p):
        limit = p * self.total()
        S = 0
        for k, v in self.items():
            S += v
            if S >= limit:
                return k

    def median(self):
        return self.quantile(0.5)

    def q1(self):
        return self.quantile(0.25)

    def q3(self):
        return self.quantile(0.75)



    # plotting methods
    @Decorators.plot
    def hist(self, **kwargs):
        plt.bar(self.keys(), self.values(), **kwargs)












if __name__ == '__main__':
    import inspect
    import sys
    import time
    from tqdm import tqdm

    import torch

    from nes_dataset import NESDataset


    ## HERE ARE REGISTERED THE UPDATING RULES

    def update_duration(stats, sequence):
        r"""Compute the duration (in seconds) of the sequence
        """
        duration = ceil(sequence[:,0,3].max().item())
        stats[duration] += 1

    def update_lengths(stats, sequence):
        r"""Compute the number of musical events in the sequence
        """
        stats[len(sequence)] += 1

    def update_num_events(stats, sequence):
        r"""Compute the number of musical events per block in the sequence
        """
        lengths = [len(block) for block in split_sequence(sequence)]
        stats.update(lengths)

    def update_voices(stats, sequence):
        r"""Compute which voices are actually present in the sequence
        """
        actual_voices = [i for i in range(sequence.shape[1]) if sequence[0,i,0] >= 0]
        stats.update(actual_voices)




    # magic trick to automatically assign names to functions
    update_rules = {
        name[7:]: func for name, func in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    }

    def split_sequence(sequence, interval=2):
        r"""Splits a sequence of events into a sequence of blocks, where each block
        contains all events for one voice during a time interval

        Args:
            sequences: torch.Tensor, shape (seq_length, num_voices, num_channels), slices indices for building blocks
                batch of musical events sequences
            interval: float (default=1)
                interval of sequences
            max_length: int
                maximal number of blocks per voice

        Returns:
            torch.Tensor, list of total_num_blocks torch.Tensors, shape (block_size, num_channels)
                sequences of blocks
        """
        all_blocks = []

        for voice in sequence.transpose(0,1):
            # compute the index of block for each event
            indices = voice[:,-1] / interval
            indices[indices < 0] = -1
            indices = indices.long()

            # compute the number of events per block
            bins, counts = torch.unique_consecutive(indices, return_counts=True)
            slices = torch.zeros(bins.max()+2, dtype=torch.int64)
            slices[bins] = counts

            # split the sequence into blocks
            blocks = list(torch.split(voice, slices.tolist()))

            blocks.pop()

            all_blocks.extend(blocks)

        return all_blocks

    # handling cli args
    argv = sys.argv[1:]
    if len(argv) == 0:
        print('Please indicate the stats you want to compute as command-line arguments.')
        print('Use -h/--help for more information about the available stats.')
        exit(1)

    if '-h' in argv or '--help' in argv:
        print('The statistics of the following quantities can be computed:')
        for name, func in update_rules.items():
            print(f'  {name}:\n    {inspect.getdoc(func)}')
        exit(0)


    dataset = NESDataset('train')
    dataset.train = False

    stats_dict = {field: Stats() for field in argv}

    for sequence in tqdm(dataset):
        for field, stats in stats_dict.items():
            update_rules[field](stats, sequence)


    for field, stats in stats_dict.items():
        print(field)
        print(stats)
        stats.hist(title=str(stats), show=True)
