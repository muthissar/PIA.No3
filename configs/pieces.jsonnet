local pieces_dict = {
  kv331_1: {
    // path: '/share/hel/home/mathias/datasets/batik_plays_mozart/midi/kv331_1.mid',
    path: '/home/mathias/Downloads/batik_plays_mozart/midi/kv331_1.mid',
    start_node: 54.5,
    n_inpaint: 32.3,
    end_window: 0,
  },
  kv332_2: {
    // path: '/share/hel/home/mathias/datasets/batik_plays_mozart/midi/kv332_2.mid',
    path: '/home/mathias/Downloads/batik_plays_mozart/midi/kv332_2.mid',
    start_node: '1:44.75',
    n_inpaint: '0:32.062',
    end_window: 0,
  },
};
function(pieces_names=std.objectFields(pieces_dict)) [pieces_dict[piece] for piece in pieces_names]
