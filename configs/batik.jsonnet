{
  local pieces_fn = import 'pieces.jsonnet',
  local ic_curve_fn = import 'ic_curves.jsonnet',
  local channel_weight_mod = import 'weights.jsonnet',
 
  local pieces = pieces_fn(['kv331_1']),
  local use_channels = ['pitch', 'time_shift'],
  local channel_weight = channel_weight_mod.get_weights('normalized_weights', use_channels),
  local ic_curve = ic_curve_fn('rampup2', channel_weight_mod.channel_idxs(use_channels)),
  app: [
    {
      local k_traces = 128,
      sampling_config: {
        k_traces: k_traces,
        temperature: 1,
        top_k: 0,
        n_poly_notes: null,
      },
      samples_per_template: 8,
      logging: 'DEBUG',
      experiment: {
        time_points_generator: {
          class_path: 'CIA.ic.FixedStepTimepoints',
          init_args: {
            step: 0.5,
            eval_step: 0.5,
            k_traces: k_traces,
            tol_placeholder_duration: 0.2,
          },
        },
        weight: {
          class_path: 'CIA.ic.MovingAverage',
          init_args: {
            window_size: 1,
            decay: 0,
            channel_weight: channel_weight,
          },
        },
        dataset: {
          class_path: 'CIA.ic.DataPiece',
          init_args: {
            label: 'batik',
            pieces: pieces,
            cache_path: '/share/hel/home/mathias/.cache/mutdata/pia/dataset_cache/BatikPlaysMozart',
          },
        },
        ic_curve: {
          class_path: 'CIA.ic.LinearInterpolation',
          init_args: ic_curve,
        },
        match_metric: 'ic',
        onset_on_next_note: true,
      },
    },
  ],
}
