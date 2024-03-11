{
  local weights = {
    normalized_weights: {
      pitch: 0.90991811,
      velocity: 0.29788502,
      duration: 0.42607584,
      time_shift: 0.68306011,
    },
    unnormalized_weights: {
      pitch: 1,
      velocity: 1,
      duration: 1,
      time_shift: 1,
    },
  },
  local channels = ['pitch', 'velocity', 'duration', 'time_shift'],
  channel_idxs: function(channels_names) [std.find(channel_name, channels)[0] for channel_name in channels_names],
  local get_weights = function(weight_type, use_channels) [if std.member(use_channels, channel) then weights[weight_type][channel] else 0.0 for channel in channels],
  mv_avg: function(window_size, decay, weight_type, use_channels) {
          class_path: 'CIA.ic.MovingAverage',
          init_args: {
            window_size: window_size,
            decay: decay,
            channel_weight: get_weights('normalized_weights', use_channels),
          },
        },
  hann: function(window_size, weight_type, use_channels) {
          class_path: 'CIA.ic.Hann',
          init_args: {
            window_size: window_size,
            channel_weight: get_weights('normalized_weights', use_channels),
          },
        },
}
