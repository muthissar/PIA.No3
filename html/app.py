import math
from typing import List
from flask import Flask, abort, make_response, render_template, redirect, send_file, url_for, jsonify, Response, request, json, g, session
from jsonargparse import ActionConfigFile, ArgumentParser
import numpy as np
from CIA.ic import DataPiece 
from ic.app import Config
import secrets
from flask import Flask

import sqlite3

# state = 'DEBUG'
state = 'RANDOM_SAMPLES'
DATABASE = 'results.db'

app = Flask(__name__) 
app.jinja_env.add_extension('jinja2.ext.do')
app.config['SECRET_KEY'] = secrets.token_hex()

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db
def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()
def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    get_db().commit()
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv
parser = ArgumentParser(
    parser_mode="jsonnet",
)
parser.add_argument("--app", type=List[Config])  
parser.add_argument("--config", action=ActionConfigFile)
config = "configs/batik.jsonnet"
args = parser.parse_args(["--config", config])
init = parser.instantiate_classes(args)
configs : List[Config] = init.app
pieces_dic = {}
# NOTE: organize by piece
for c in configs:
    out_folder_1 = c.out
    dataset = c.experiment.dataset
    if not isinstance(dataset, DataPiece):
        raise NotImplementedError("Only DataPiece is supported")
    dataset : DataPiece = dataset
    for piece in dataset.pieces:
        piece_name = piece.name
        folder = out_folder_1.joinpath(piece_name)
        assert folder.exists()
        piece_dic = pieces_dic.get(piece_name, {})
        piece_dic[str(c.experiment.hash_name)] = folder
        pieces_dic[piece_name] = piece_dic

def set_state(chosen_exp, piece_name_, all_sample_ids_):
    global piano_roll_gen
    global used_template
    global audio_file
    global templates
    global chosen
    global sorted_names
    global piece_name
    global all_sample_ids
    # session['piano_roll_gen']
    # session['used_template']
    # session['audio_file']
    # session['templates']
    # session['chosen']
    # session['sorted_names']
    # session['piece_name']
    # session['all_sample_ids']

    all_sample_ids = all_sample_ids_
    chosen = chosen_exp
    piece_name = piece_name_
    piece = pieces_dic[piece_name]
    # NOTE: we always sort the whole list
    sorted_names = list(sorted(piece.keys()))
    # TODO: here we use both the sorted names (for deterministically choosing between experiemtns. We could change everywhere this to use ids.)
    chosen_id = sorted_names[chosen]
    chosen_curve = piece[chosen_id]
    all_curves = [piece[curve_name] for curve_name in sorted_names]
    
    chosen_sample_folder = chosen_curve.joinpath(str(all_sample_ids[sorted_names.index(chosen_id)]))
    all_sample_folders = [p.joinpath(str(i)) for i, p in zip(all_sample_ids, all_curves)]
    chosen_plot_folder = chosen_sample_folder.joinpath('plotly_figs')
    
    piano_roll_gen = chosen_plot_folder.joinpath("piano_roll_1.json")
    used_template = chosen_plot_folder.joinpath("ic_int_summed_1.json")
    audio_file = chosen_sample_folder.joinpath("song.mp3")
    templates = [p.joinpath('plotly_figs', "match_ic_int_summed.json") for p in all_sample_folders]
    
piece_name = 'kv331_1_start_-45_nodes_134_end_0'
chosen = 0 # '0e3ee2b957cff8903c6c1eb79457a6add2cf5c2c8e88c94855ec5f6faa0b4f35'
# chosen = 1 # '3bceac27346acddd96b11d908f3ac92fd667561f0f2fad7622b5b3d5a4b6d47e'
# chosen = 2 # '80f8da596bf413a11494bedba1c1793778e648bda3bc2d2c1a49b7e2cca22669'
all_sample_ids = [str(0) for _ in range(len(configs))]
set_state(chosen, piece_name, all_sample_ids)

@app.route("/") 
def home(): 
    if state == 'DEBUG':
        return render_template("debug.html", n_curves=len(templates))
    elif state == 'RANDOM_SAMPLES':
        return render_template("random_samples.html", n_curves=len(templates))
    else:
        return render_template("main.html", n_curves=len(templates))
 
@app.route("/curve/<int:curve_id>")
def curve(curve_id: int):
    curve = templates[curve_id]
    return send_file(curve.absolute(), as_attachment=False, max_age=0)
@app.route("/choose", methods=["POST"]) 
def choose():
    choice = request.form.get('choice', "")
    success = False
    with app.app_context():
        if choice and choice.isdigit():
            correct_exp = sorted_names[chosen]
            correct_sample = all_sample_ids[chosen]
            choice = int(choice)
            selected_exp = sorted_names[choice]
            selected_sample = all_sample_ids[choice]
            res = query_db("INSERT INTO result (piece, correct_exp, correct_sample, selected_exp, selected_sample ) VALUES (?,?,?,?,?)", (piece_name, correct_exp,correct_sample,selected_exp,selected_sample))
            if chosen == choice:
                success = True
    with open(used_template, 'r') as f:
        template = json.load(f)
    return make_response(jsonify({"success": success, "correct": chosen, "generated_curve": template}), 200)
@app.route("/audio")
def audio():
    return send_file(audio_file.absolute(), as_attachment=True, max_age=0)
@app.route("/piano_roll")
def piano_roll():
    return send_file(piano_roll_gen.absolute(), as_attachment=False, max_age=0)
  
@app.route("/template")
def template():
    if state == "DEBUG":
        return send_file(used_template.absolute(), as_attachment=False, max_age=0)
    else:
        abort(403)

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
@app.route("/debug", methods=["POST"])
def debug():
    if state == "DEBUG":
        try:
            piece_name_ = request.form.get('piece_name')
            chosen = int(request.form.get('chosen_curve_id'))
            all_sample_ids = json.loads(request.form.get('all_sample_ids'))
            for i in all_sample_ids:
                if not isinstance(i, int):
                    raise ValueError(f"Expected string, got {i}")
        except Exception as e:
            # return make_response(jsonify({"success": False, "error": str(e)}), 500)
            return make_response(jsonify({"success": False}), 500)
    elif state == "RANDOM_SAMPLES":
        chosen = np.random.randint(0, len(configs))
        # TODO generalize
        all_sample_ids = np.random.randint(0, 8, len(configs))
        piece_name_ = piece_name
    set_state(chosen, piece_name_, all_sample_ids)
    return redirect(url_for('home'))
if __name__ == "__main__": 
    app.run(debug=False) 