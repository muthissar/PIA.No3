from typing import List
from flask import Flask, abort, make_response, render_template, redirect, send_file, url_for, jsonify, Response, request, json
from jsonargparse import ActionConfigFile, ArgumentParser
from CIA.ic import DataPiece 
from ic.app import Config


DEBUG = True

app = Flask(__name__) 
app.jinja_env.add_extension('jinja2.ext.do')
parser = ArgumentParser(
    parser_mode="jsonnet",
)
parser.add_argument("--app", type=List[Config])  
parser.add_argument("--config", action=ActionConfigFile)
config = "configs/batik.jsonnet"
args = parser.parse_args(["--config", config])
init = parser.instantiate_classes(args)
configs : List[Config] = init.app
# NOTE: find the plots
# app[0].out.rglob('**/plotly_figs/')
# NOTE: for now we hardcode it
pieces_dic = {}
# NOTE: organize by piece
for c1 in configs:
    out_folder_1 = c1.out
    dataset = c1.experiment.dataset
    if not isinstance(dataset, DataPiece):
        raise NotImplementedError("Only DataPiece is supported")
    dataset : DataPiece = dataset
    for piece in dataset.pieces:
        piece_name = piece.name
        folder = out_folder_1.joinpath(piece_name)
        assert folder.exists()
        piece_arr = pieces_dic.get(piece_name, [])
        piece_arr.append(folder)
        pieces_dic[piece_name] = piece_arr

def set_state(chosen_, piece_name, chosen_curve_id, all_sample_ids):
    global chosen
    global piano_roll_gen
    global used_template
    global audio_file
    global templates
    chosen = chosen_
    piece = pieces_dic[piece_name]
    chosen_curve = piece[chosen_curve_id]
    all_curves = [p for p in piece]
# all_sample_ids = [str(0) for _ in all_curves]

# TODO: generalize
# sample_folder = Path('out/80f8da596bf413a11494bedba1c1793778e648bda3bc2d2c1a49b7e2cca22669/samplingconfig-k-traces-128-temperature-1-0-n-poly-notes-none-dynamic-temperature-max-ic-none-top-p-0-0-top-k-0/kv332_2_start_149_nodes_177_end_0/5')
    chosen_sample_folder = chosen_curve.joinpath(str(all_sample_ids[chosen]))
# TODO: generalize

    all_sample_folders = [p.joinpath(str(i)) for i, p in zip(all_sample_ids, all_curves)]

    chosen_plot_folder = chosen_sample_folder.joinpath('plotly_figs')
    piano_roll_gen = chosen_plot_folder.joinpath("piano_roll_1.json")
    used_template = chosen_plot_folder.joinpath("ic_int_summed_1.json")
    audio_file = chosen_sample_folder.joinpath("song.mp3")
    # return piano_roll_gen,used_template,audio_file,all_sample_folders

    
    templates = [p.joinpath('plotly_figs', "match_ic_int_summed.json") for p in all_sample_folders]
    #piano_roll_gen, used_template, audio_file, all_sample_folders = set_state(chosen, pieces_dic, piece_name, chosen_curve_id, all_sample_ids)
# TODO: generalize
piece_name = 'kv331_1_start_-45_nodes_134_end_0'
chosen_curve_id = 0
chosen = 0
all_sample_ids = [str(0) for _ in range(len(configs))]
set_state(chosen, piece_name, chosen_curve_id, all_sample_ids)

@app.route("/") 
def home(): 
    # return render_template("index.html") 
    if DEBUG:
        return render_template("debug.html", n_curves=len(templates))
    else:
        return render_template("main.html", n_curves=len(templates))
 
@app.route("/curve/<int:curve_id>")
def curve(curve_id: int):
    curve = templates[curve_id]
    # with open(curve, 'r') as f:
    #     data = json.load(f)
    #     return jsonify(data)
    return send_file(curve.absolute(), as_attachment=False)
@app.route("/choose", methods=["POST"]) 
def choose():
    choice = request.form.get('choice', "")
    if choice and choice.isdigit() and chosen == int(choice):
        success = True
    else:
        success = False
    return make_response(jsonify({"success": success}), 200)
@app.route("/audio")
def audio():
    return send_file(audio_file.absolute(), as_attachment=True)
@app.route("/piano_roll")
def piano_roll():
    return send_file(piano_roll_gen.absolute(), as_attachment=False)
  
@app.route("/template")
def template():
    if DEBUG:
        return send_file(used_template.absolute(), as_attachment=False)
    else:
        abort(403)
if DEBUG:
    @app.route("/debug", methods=["POST"])
    def debug():
        try:
            piece_name = request.form.get('piece_name')
            chosen_curve_id = int(request.form.get('chosen_curve_id'))
            chosen = int(request.form.get('chosen'))
            all_sample_ids = json.loads(request.form.get('all_sample_ids'))
            for i in all_sample_ids:
                if not isinstance(i, int):
                    raise ValueError(f"Expected string, got {i}")
            set_state(chosen, piece_name, chosen_curve_id, all_sample_ids)
            return redirect(url_for('home'))
        except Exception as e:
            # return make_response(jsonify({"success": False, "error": str(e)}), 500)
            return make_response(jsonify({"success": False}), 500)
if __name__ == "__main__": 
    app.run(debug=False) 