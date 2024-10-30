{
  local channel_weight_mod = import 'weights.jsonnet',
  local use_channels = ['pitch', 'time_shift'],
  local weight = channel_weight_mod.hann(2.0, 'normalized_weights', use_channels),
  local k_traces = 128,
  app:
    {
        sampling_config: {
            dynamic_temperature_max_ic: 50,
            k_traces: k_traces,
            temperature: 1,
            top_k: 0,
            n_poly_notes: null,
        },
        experiment: {
            weight: weight,
                time_points_generator: {
                class_path: 'ic.beam_search.timepoints.FixedStepTimepoints',
                init_args: {
                    step: 0.3,
                    eval_step: 0.1,
                    k_traces: k_traces,
                    tol_placeholder_duration: 0.2,
                },
            },
            match_metric: 'ic',
            onset_on_next_note: true,
        }
    },
}
