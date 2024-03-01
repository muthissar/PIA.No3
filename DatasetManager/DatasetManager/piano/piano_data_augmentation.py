import math
import random


def time_shift(x, sequence_size, hop_size):
    time_shift = math.floor(random.uniform(0, hop_size))
    x = x[time_shift:time_shift + sequence_size]
    return x


def time_dilation(x, time_dilation_factor):
    for t in range(len(x)):
        # time_shift
        old_ts = x[t, ts_pos]
        ts_value = self.index2value['time_shift'][old_ts]
        if ts_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]:
            continue
        ts_dilated = find_nearest_value(self.time_table, ts_value * time_dilation_factor)
        new_ts = self.value2index['time_shift'][ts_dilated]
        x[t, ts_pos] = new_ts

        #  duration
        old_duration = x[t, duration_pos]
        duration_value = self.index2value['duration'][old_duration]
        if duration_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]:
            continue
        duration_dilated = find_nearest_value(self.time_table, duration_value * time_dilation_factor)
        if duration_dilated == 0.0:
            # smallest duration
            new_duration = 0
        else:
            new_duration = self.value2index['duration'][duration_dilated]
        x[t, duration_pos] = new_duration


def transposition(self, x):
    if self.transformations['transposition']:
        # Draw a random transposition
        transposition = int(random.uniform(-self.max_transposition, self.max_transposition))

        if transposition != 0:
            x_trans = x
            for t in range(len(x)):
                # pitch
                old_pitch = x[t, pitch_pos]
                pitch_value = self.index2value['pitch'][old_pitch]
                if pitch_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]:
                    continue
                pitch_transposed = pitch_value + transposition
                if pitch_transposed not in self.value2index['pitch'].keys():
                    # Transposition not allowed for that chunk... don't transpose then
                    x_trans = x
                    break
                new_pitch = self.value2index['pitch'][pitch_transposed]
                x_trans[t, pitch_pos] = new_pitch
            x = x_trans

    return x, time_dilation_factor, velocity_shift