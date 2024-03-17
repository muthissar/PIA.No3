{
  local pieces_fn = import 'pieces.jsonnet',
  local channel_weight_mod = import 'weights.jsonnet',
  local use_channels = ['pitch', 'time_shift'],
  // local channel_weight = channel_weight_mod.get_weights('normalized_weights', use_channels),
  // local weight = channel_weight_mod.mv_avg(1, 0, 'normalized_weights', use_channels),
  // local use_channels_arr = [['pitch', 'time_shift'], ['pitch'], ['time_shift']],
  local weight = channel_weight_mod.hann(2.0, 'normalized_weights', use_channels),
  local k_traces_arr = [1, 2, 4, 16, 64, 128],
  // local k_traces_arr = [128],
  local step_arr = [0.1, 0.2, 0.5, 1.0, 2.0],
  local eval_step = 0.1,
  // local step = [0.5]
  // local eval_step = .5
  local BaseConfig = import 'base_config.jsonnet',
  local config = BaseConfig(16) +
  {
    dataset: {
      class_path: 'ic.ic.DataCache',
      init_args: {
        label: 'batik',
        split: 'validation',
        // NOTE: 10.0 get's falsely parsed as an integer
        n_inpaint: 10.00001,
        n_pieces: 500,
        end_window: 0.0,
        midi_path: '/share/hel/home/mathias/datasets/batik_plays_mozart',
        cache_path: '/share/hel/home/mathias/.cache/mutdata/pia/dataset_cache/BatikPlaysMozart',
      },
    },
    weight: weight,
  },
  app:
    [config { experiment+: { time_points_generator+: { init_args+: { eval_step: eval_step, step: step } } } } for step in step_arr],
}
// k_traces,
// steps
// temperature
