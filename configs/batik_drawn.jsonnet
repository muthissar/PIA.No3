{
  local pieces_fn = import 'pieces.jsonnet',
  local ic_curve_fn = import 'ic_curves.jsonnet',
  local channel_weight_mod = import 'weights.jsonnet',
  local BaseConfig = import 'base_config.jsonnet',
  // local pieces = pieces_fn(['kv331_1']),
  local pieces = pieces_fn(),
  local use_channels = ['pitch', 'time_shift'],
  // local channel_weight = channel_weight_mod.get_weights('normalized_weights', use_channels),
  // local weight = channel_weight_mod.mv_avg(1, 0, 'normalized_weights', use_channels),
  local weight = channel_weight_mod.hann(2.0, 'normalized_weights', use_channels),
  // local ic_curve = ic_curve_fn('rampup2', channel_weight_mod.channel_idxs(use_channels)),
  local k_traces = 128,
  // local curves = [
  //   'square2constant2',
  //   'rampup2constantlow2',
  //   'rampdown2constantlow2',
  //   'constantlow',
  //   'constanthigh',
  //   'rampup2',
  //   'rampdown2',
  //   'square2constant2shifted',
  //   'rampup2constantlow2shifted',
  //   'rampdown2constantlow2shifted',
  // ],
  local curves = [
    'constantlow',
    'constanthigh',
    'rampup2',
    'rampdown2',
    'square',
    'squareoffset'
  ],
  local ic_curves = [ic_curve_fn(curve, channel_weight_mod.channel_idxs(use_channels)) for curve in curves],
  app:
    [
      BaseConfig(128) +
        {
          experiment+: {
              dataset: {
                class_path: 'ic.data.DataPiece',
                init_args: {
                  label: 'batik',
                  pieces: pieces,
                  cache_path: std.extVar('CACHE_DIR')+'/dataset_cache/BatikPlaysMozart',
                },
              },
              weight: weight,
              ic_curve: {
                class_path: 'ic.curves.LinearInterpolation',
                init_args: ic_curve,
              },
            
          },
          samples_per_template: 8,
        } for ic_curve in ic_curves
    ],
}
