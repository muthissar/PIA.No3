{
  local pieces_fn = import 'pieces.jsonnet',
  local channel_weight_mod = import 'weights.jsonnet',
  local env = import 'env.jsonnet',
  local use_channels = ['pitch', 'time_shift'],
  // local channel_weight = channel_weight_mod.get_weights('normalized_weights', use_channels),
  // local weight = channel_weight_mod.mv_avg(1, 0, 'normalized_weights', use_channels),
  // local use_channels_arr = [['pitch', 'time_shift'], ['pitch'], ['time_shift']],
  local weight = channel_weight_mod.hann(2.0, 'normalized_weights', use_channels),
  local k_traces_arr = [1, 2, 4, 8, 16, 24, 32, 64, 96, 128],
  local dynamic_temperature_max_ic_arr = [10, 20, 30, 40, 50, 60, 70, 80, 120],
  // local dynamic_temperature_max_ic_arr = [30,],
  // local k_traces_arr = [128],
  local step_arr = [0.4, 0.5, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1 , 1.0, 2.0],
  local eval_step = 0.1,
  local n_pieces = 400,
  // local step = [0.5]
  // local eval_step = .5
  local BaseConfig = import 'base_config.jsonnet',
  local data = {
    experiment+: {
      dataset: {
        class_path: 'ic.data.DataCache',
        init_args: {
          label: 'batik',
          split: 'validation',
          // NOTE: 10.0 get's falsely parsed as an integer
          n_inpaint: 10.00001,
          n_pieces: n_pieces,
          end_window: 0.0,
          // midi_path: std.extVar('BATIK_DIR'),
          // cache_path: std.extVar('CACHE_DIR') + '/dataset_cache/BatikPlaysMozart',
          midi_path: env.BATIK_DIR,
          cache_path: env.CACHE_DIR,
        },
      },
      weight: weight,
    },
  },
  app:
    // NOTE: BEST
    [BaseConfig(128) + data + { experiment+: { time_points_generator+: { init_args+: {step: 0.3 } } }, sampling_config+: { dynamic_temperature_max_ic: 50 }  }] +
    // NOTE: Ablations....
    [BaseConfig(16) + data + { sampling_config+: { dynamic_temperature_max_ic: dynamic_temperature_max_ic } } for dynamic_temperature_max_ic in dynamic_temperature_max_ic_arr] +
    [BaseConfig(16) + data + { experiment+: { time_points_generator+: { init_args+: { step: step } } } } for step in step_arr] + 
    [BaseConfig(k_trace) + data for k_trace in k_traces_arr]
}
// k_traces,
// steps
// temperature
