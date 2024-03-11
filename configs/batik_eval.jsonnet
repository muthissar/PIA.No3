{
  local pieces_fn = import 'pieces.jsonnet',
  local ic_curve_fn = import 'ic_curves.jsonnet',
  local channel_weight_mod = import 'weights.jsonnet',
  local use_channels = ['pitch', 'time_shift'],
  // local channel_weight = channel_weight_mod.get_weights('normalized_weights', use_channels),
  // local weight = channel_weight_mod.mv_avg(1, 0, 'normalized_weights', use_channels),
  local weight = channel_weight_mod.hann(2.0, 'normalized_weights', use_channels),
  local k_traces_arr = [1, 2, 4, 16, 64, 128],
  // local k_traces_arr = [128],
  local step_arr = [0.1, 0.2 , 0.5, 1.0, 2.0],
  local eval_step = 0.1,
  // local step = [0.5]
  // local eval_step = .5
  local BaseConfig = 
    {
      local k_traces = 16,
      sampling_config: {
        k_traces: k_traces,
        temperature: 1,
        top_k: 0,
        n_poly_notes: null,
      },
      samples_per_template: 4,
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
        weight: weight,
        // {
        //   class_path: 'CIA.ic.MovingAverage',
        //   init_args: {
        //     window_size: 1,
        //     decay: 0,
        //     channel_weight: channel_weight,
        //   },
        // },
        dataset: {
          class_path: 'CIA.ic.DataCache',
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
        match_metric: 'ic',
        onset_on_next_note: true,
      },
    },
  app: 
  // [BaseConfig + {sampling_config+: {k_traces: k_traces}, experiment+: {time_points_generator+: {init_args+: {k_traces: k_traces}}}} for k_traces in k_traces_arr] + 
  // []
  [BaseConfig + {experiment+: {time_points_generator+: {init_args+: {eval_step: eval_step, step: step}}}} for step in step_arr]
}
// k_traces,
// steps
// temperature