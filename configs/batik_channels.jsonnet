{
  local pieces_fn = import 'pieces.jsonnet',
  local channel_weight_mod = import 'weights.jsonnet',
  local use_channels_arr = [['pitch'], ['time_shift'],['pitch', 'time_shift']],
  local BaseConfig = import 'base_config.jsonnet',
  app:
    [BaseConfig(16) +
     {
       experiment+: {
         dataset: {
           class_path: 'ic.data.DataCache',
           init_args: {
             label: 'batik',
             split: 'validation',
             // NOTE: 10.0 get's falsely parsed as an integer
             n_inpaint: 10.00001,
             // n_pieces: 500,
             n_pieces: 10,
             end_window: 0.0,
             midi_path: '/share/hel/home/mathias/datasets/batik_plays_mozart',
             cache_path: '/share/hel/home/mathias/.cache/mutdata/pia/dataset_cache/BatikPlaysMozart',
           },
         },
         // weight: channel_weight_mod.mv_avg(1, 0, 'normalized_weights', use_channels),
         weight: channel_weight_mod.hann(2.0, 'normalized_weights', use_channels),
       },
       samples_per_template: 4,
     } for use_channels in use_channels_arr],
}