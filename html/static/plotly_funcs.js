plot_div_id = "plot"
function fetch_curves(curves) {
  return Promise.all(curves.map(curve => {
    return fetch(curve)
      .then(response => response.json())
      .catch(
        error => console.error('Error:', error)
      );
  }))
}
function draw_curves(curves) {
  fetch_curves(curves).then((ploty_curves) => {
    for (i = 0; i < ploty_curves.length; i++) {

      plotly_fig = ploty_curves[i]
      // Use Plotly to plot the data
      data_plot = plotly_fig.data[0];

      if (data_plot['type'] == 'heatmap') {
        ys = [0, data_plot.z.length]
        data_plot.coloraxis = {
          showscale: false,
          colorscale: 'Greens',

        }
        data_plot.showscale = false
        data_plot.colorscale = 'Hot'
        data_plot.reversescale = true
        plotly_fig.data = [data_plot]
      }
      else {
        ys = [Math.min(...data_plot.y), Math.max(...data_plot.y)]
      }
      delete data_plot.xaxis
      data_plot.yaxis = `y${i + 1}`
      xs = [Math.min(...data_plot.x), Math.min(...data_plot.x)]
      var trace1 = {
        x: xs,
        y: ys,
        mode: 'lines+markers',
        type: 'scatter',
        name: 'Trace 1',
        yaxis: `y${i + 1}`
      };
      plotly_fig.data.push(trace1)
      plotly_fig.layout['showlegend'] = false;
    }
    var navbar = document.getElementsByTagName('nav')[0];  // Replace 'navbar' with the id of your navbar
    var navbarHeight = navbar.offsetHeight;
    var layout = {
      // height: 500,
      autosize: true,  // This makes the plot resize to fit the window
      width: window.innerWidth,
      height: window.innerHeight - navbarHeight,
      showlegend: false,
      showscale: false,
      coloraxis_showscale: false,
      margin:{
        // l: 0,
        // r: 0,
        b: 0,
        t: 0,
        // pad: 0
      },
      annotations: [

      ],
    };
    // NOTE: keep an extra for the template...
    num_figs = ploty_curves.length + 1
    margin_sub_figs = 1 / num_figs * 0.25
    for (i = 0; i < num_figs; i++) {
      layout[`yaxis${num_figs - i}`] = {
        domain: [i / num_figs, (i + 1) / num_figs - margin_sub_figs]
      }
      if (i == num_figs - 1) {
        layout[`yaxis${num_figs - i}`].range = [24, 96]
        title = 'template curve';
        title_visible = false
      } else {
        layout[`yaxis${num_figs - i}`].range = [0, 25]
        title = curves[i];
        title_visible = true
      }
      
      layout.annotations.push({
        text: title,
        visible: title_visible,
        font: {
          size: 20,
          color: 'black',
        },
        showarrow: false,
        align: 'center',
        x: .5,
        y: 1-i/num_figs-1*margin_sub_figs/2,
        xref: 'paper',
        yref: 'paper',
      })
    }
    data = ploty_curves.reduce((accumulator, currentValue) => {
      return accumulator.concat(currentValue.data);
    }, []);
    Plotly.newPlot(plot_div_id, data, layout);

    var audio_div = document.querySelector('audio');
    var plot_div = document.getElementById(plot_div_id);
    plot_div.on('plotly_click', function (data) {
      audio_div.currentTime = data.points[0].x;
    });

  });
};

function add_form_script() {
  const form = document.getElementById('choose-form');
  form.addEventListener('submit', function (event) {
    event.preventDefault();  // Prevent the form from submitting normally
    fetch('choose', {
      method: 'POST',
      body: new FormData(form),
    })
      .then(response => response.json())
      .then(result => {
        if (result.success) {
          document.getElementById('successAlert').style.display = 'inline-block';
        }
        else {
          fail_alert = document.getElementById('failAlert')
          fail_alert.style.display = 'inline-block';
          fail_alert.textContent = `Wrong curve picked, the correct curve was: ${result.correct}`;
        }
        template = result.generated_curve.data[0]
        delete template.xaxis
        const select = form.getElementsByTagName('select')[0];
        // TODO: change this not to depend on number of elements in html.
        template_index = select.options.length
        template.yaxis = `y${template_index +1}`
        Plotly.addTraces(plot_div_id, template)
        update = {}
        update[`annotations[${template_index}].visible`] = true
        Plotly.relayout(plot_div_id, update)
        form.getElementsByTagName('button')[0].disabled = true
        form.getElementsByTagName('select')[0].disabled = true
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  });
}

function range(start, stop, step) {
  if (typeof stop == 'undefined') {
    // one param defined
    stop = start;
    start = 0;
  }

  if (typeof step == 'undefined') {
    step = 1;
  }

  if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
    return [];
  }

  var result = [];
  for (var i = start; step > 0 ? i < stop : i > stop; i += step) {
    result.push(i);
  }

  return result;
};

function get_animate_curves(time_cursable) {
  return function (e) {
    prog = e.currentTime;
    Plotly.restyle(plot_div_id, update = { 'x': [[prog, prog]] }, traceIndices = range(0, time_cursable.length).map(e => 2 * e + 1));
  }
}