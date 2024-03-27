{
  local pieces_fn = import 'pieces.jsonnet',
  // local ic_curve_fn = import 'ic_curves_high.jsonnet',
  local ic_curve_fn = import 'ic_curves_func.jsonnet',
  local channel_weight_mod = import 'weights.jsonnet',
  local BaseConfig = import 'base_config.jsonnet',
  local env = import 'env.jsonnet',
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
  // local curves = [
  //   // ['constantlow', [_, _]],
  //   ['constanthigh', null, 10.0],
  //   ['constanthigh', null, 7.5],
  //   ['constanthigh', null, 5],
  //   // 'rampup2',
  //   // 'rampdown2',
  //   // 'square',
  //   // 'squareoffset'
  // ],
    local curves = [
    // ['constantlow', null, null],
    // ['constanthigh', null, null],
    ['constantlow', 5.0, 5.0],
    ['rampdown2', null, null],
    ['squareoffset', null, null],
    ['rampup2', null, null],
    ['square', null, null],
    // ['constanthigh', 10.0, 10.0],
  ],
  local ic_curves = [ic_curve_fn(curve=curve[0], channels=channel_weight_mod.channel_idxs(use_channels), low=curve[1], high=curve[2]) for curve in curves],
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
                  // cache_path: std.extVar('CACHE_DIR')+'/dataset_cache/BatikPlaysMozart',
                  cache_path: env.CACHE_DIR+'/dataset_cache/BatikPlaysMozart',
                },
              },
              weight: weight,
              ic_curve: {
                class_path: 'ic.curves.LinearInterpolation',
                init_args: ic_curve,
              },
              time_points_generator+: {
                init_args+: {
                  step: 0.3,
                }
              }
            
          },
          samples_per_template: 144,
        } +
        { 
          sampling_config+: {
            dynamic_temperature_max_ic: 50,
          }
        }
      
      for ic_curve in ic_curves
    ],
}
