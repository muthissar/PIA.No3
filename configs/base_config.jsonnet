function (k_traces) 
{
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
      class_path: 'ic.beam_search.timepoints.FixedStepTimepoints',
      init_args: {
        // step: 0.5,
        step: 0.3,
        eval_step: 0.1,
        k_traces: k_traces,
        tol_placeholder_duration: 0.2,
      },
    },
    match_metric: 'ic',
    onset_on_next_note: true,
  },
}
