local ic_curves = {
  rampup2:
    {
      relative_time: true,
      timepoints: [0, 0.499, 0.5, 1],
      ics: [
        [10, 10, 10, 10],
        [0, 0, 0, 0],
        [10, 10, 10, 10],
        [0, 0, 0, 0],
      ],
    },
  rampdown2:
    {
      relative_time: true,
      timepoints: [0, 0.499, 0.5, 1],
      ics: [
        [0, 0, 0, 0],
        [10, 10, 10, 10],
        [0, 0, 0, 0],
        [10, 10, 10, 10],
      ],
    },
  square2:
    {
      relative_time: true,
      timepoints: [0, 0.249, 0.25, 0.499, 0.5, 0.749, 7.5, 1],
      ics: [
        [10, 10, 10, 10],
        [10, 10, 10, 10],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [10, 10, 10, 10],
        [10, 10, 10, 10],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
    },
};
function (curve, channels=[0,1,2,3]) ic_curves[curve] + {ics: [[if std.member(channels, j) then ics[j] else 0.0 for j in std.range(0, 3)] for ics in ic_curves[curve].ics]}
