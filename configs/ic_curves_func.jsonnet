local ic_curves = {
  rampdown2: function(low_=0.0, high_=12.5)
    {
      local low = if low_ == null then 0.0 else low_,
      local high = if high_ == null then  12.5 else high_,
      relative_time: true,
      timepoints: [0, 0.499, 0.5, 1],
      ics: [
        [high, high, high, high],
        [low, low, low, low],
        [high, high, high, high],
        [low, low, low, low],
      ],
    },
  rampup2: function(low_=0.0, high_=12.5)
    {
      local low = if low_ == null then 0.0 else low_,
      local high = if high_ == null then  12.5 else high_,
      relative_time: true,
      timepoints: [0, 0.499, 0.5, 1],
      ics: [
        [low, low, low, low],
        [high, high, high, high],
        [low, low, low, low],
        [high, high, high, high],
      ],
    },
  square: function(low_=1.5, high_=10.0)
    {
      local low = if low_ == null then 1.5 else low_,
      local high = if high_ == null then  10.0 else high_,
      relative_time: true,
      timepoints: [0, 0.499, 0.5, 1],
      ics: [
        [high, high, high, high],
        [high, high, high, high],
        [low, low, low, low],
        [low, low, low, low],
      ],
    },
  squareoffset: function(low_=1.5, high_=10.0)
    {
      local low = if low_ == null then 1.5 else low_,
      local high = if high_ == null then 10.0 else high_,
      relative_time: true,
      timepoints: [0, 0.499, 0.5, 1],
      ics: [
        [low, low, low, low],
        [low, low, low, low],
        [high, high, high, high],
        [high, high, high, high],
      ],
    },
  square2:
    {
      relative_time: true,
      timepoints: [0, 0.249, 0.25, 0.499, 0.5, 0.749, 0.75, 1],
      ics: [
        [10, 10, 10, 10],
        [10, 10, 10, 10],
        [1.5, 1.5, 1.5, 1.5],
        [1.5, 1.5, 1.5, 1.5],
        [10, 10, 10, 10],
        [10, 10, 10, 10],
        [1.5, 1.5, 1.5, 1.5],
        [1.5, 1.5, 1.5, 1.5],
      ],
    },
  square2constant:
    {
      relative_time: true,
      timepoints: [0, 0.249, 0.25, 0.499, 0.5, 0.749, 0.75, 1],
      ics: [
        [12.5, 12.5, 12.5, 12.5],
        [12.5, 12.5, 12.5, 12.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [12.5, 12.5, 12.5, 12.5],
        [12.5, 12.5, 12.5, 12.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
      ],
    },
  square2constant2:
    {
      relative_time: true,
      timepoints: [0, 0.249, 0.25, 0.499, 0.5, 0.749, 0.75, 1],
      ics: [
        [10, 10, 10, 10],
        [10, 10, 10, 10],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [10, 10, 10, 10],
        [10, 10, 10, 10],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
      ],
    },
  square2constant2shifted:
    {
      relative_time: true,
      timepoints: [0, 0.249, 0.25, 0.499, 0.5, 0.749, 0.75, 1],
      ics: [
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [10, 10, 10, 10],
        [10, 10, 10, 10],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [10, 10, 10, 10],
        [10, 10, 10, 10],
      ],
    },
  rampdown2pause:
    {
      relative_time: true,
      timepoints: [0, 0.249, 0.25, 0.499, 0.5, 0.749, 0.75, 1],
      ics: [
        [10, 10, 10, 10],
        [1.5, 1.5, 1.5, 1.5],
        [1.5, 1.5, 1.5, 1.5],
        [1.5, 1.5, 1.5, 1.5],
        [10, 10, 10, 10],
        [1.5, 1.5, 1.5, 1.5],
        [1.5, 1.5, 1.5, 1.5],
        [1.5, 1.5, 1.5, 1.5],
      ],
    },
  rampdown2constantlow:
    {
      relative_time: true,
      timepoints: [0, 0.249, 0.25, 0.499, 0.5, 0.749, 0.75, 1],
      ics: [
        [10, 10, 10, 10],
        [5, 5, 5, 5],
        [5, 5, 5, 5],
        [5, 5, 5, 5],
        [10, 10, 10, 10],
        [5, 5, 5, 5],
        [5, 5, 5, 5],
        [5, 5, 5, 5],
      ],
    },
  rampdown2constantlow2:
    {
      relative_time: true,
      timepoints: [0, 0.249, 0.25, 0.499, 0.5, 0.749, 0.75, 1],
      ics: [
        [12.5, 12.5, 12.5, 12.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [12.5, 12.5, 12.5, 12.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
      ],
    },
  rampdown2constantlow2shifted:
    {
      relative_time: true,
      timepoints: [0, 0.249, 0.25, 0.499, 0.5, 0.749, 0.75, 1],
      ics: [
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [12.5, 12.5, 12.5, 12.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [12.5, 12.5, 12.5, 12.5],
        [2.5, 2.5, 2.5, 2.5],
      ],
    },
  rampup2constantlow2:
    {
      relative_time: true,
      timepoints: [0, 0.249, 0.25, 0.499, 0.5, 0.749, 0.75, 1],
      ics: [
        [2.5, 2.5, 2.5, 2.5],
        [12.5, 12.5, 12.5, 12.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [12.5, 12.5, 12.5, 12.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
      ],
    },
  rampup2constantlow2shifted:
    {
      relative_time: true,
      timepoints: [0, 0.249, 0.25, 0.499, 0.5, 0.749, 0.75, 1],
      ics: [
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [12.5, 12.5, 12.5, 12.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5, 2.5],
        [12.5, 12.5, 12.5, 12.5],
      ],
    },
  constantlow: function(low_=2.5, high_=-1)
    {
      local low = if low_ == null then 2.5 else low_,
      local high = if high_ == null then -1 else high_,
      relative_time: true,
      timepoints: [0, 1],
      ics: [
        [low, low, low, low],
        [low, low, low, low],
      ],
    },
  constanthigh: function(low_=-1, high_=10.0)
    {
      local low = if low_ == null then -1 else low_,
      local high = if high_ == null then 10.0 else high_,
      relative_time: true,
      timepoints: [0, 1],
      ics: [
        [high, high, high, high],
        [high, high, high, high],
      ],
    },

};
function(curve, channels=[0, 1, 2, 3], low=null, high=null) ic_curves[curve](low, high) + {
  ics: [
    [if std.member(channels, j) then ics[j] else 0.0 for j in std.range(0, 3)]
    for ics in ic_curves[curve](low, high).ics
  ],
}
