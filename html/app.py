import hashlib

import os
import pickle
from typing import List
from pathlib import Path
from flask import Flask, abort, make_response, render_template, redirect, send_file, url_for, jsonify, Response, request, json, g, session
from jsonargparse import ActionConfigFile, ArgumentParser
import numpy as np
from CIA.ic import DataPiece 
from ic.app import Config
import secrets
from flask import Flask
import datetime

import sqlite3

# state = 'DEBUG'
state = 'RANDOM_SAMPLES'
DATABASE = 'results.db'

app = Flask(__name__) 
app.jinja_env.add_extension('jinja2.ext.do')
# app.config['SECRET_KEY'] = secrets.token_hex()
app.secret_key = 'ec660d492e2f13be02d7519c77ef0770a8f1a741d11562e79763cbd6f46ee7a3'

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
file_name = 'configs.pkl'

# Check if the file exists
if os.path.exists(file_name):
    # Load the configurations from the file
    with open(file_name, 'rb') as f:
        configs = pickle.load(f)
else:
    # Save the configurations to the file
    with open(file_name, 'wb') as f:
        parser = ArgumentParser(
            parser_mode="jsonnet",
        )
        parser.add_argument("--app", type=List[Config])  
        parser.add_argument("--config", action=ActionConfigFile)
        config = "configs/batik.jsonnet"
        args = parser.parse_args(["--config", config])
        init = parser.instantiate_classes(args)
        configs : List[Config] = init.app
        pickle.dump(configs, f)
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

def set_state(chosen_exp, piece_name_, all_sample_ids):
    session['all_sample_ids'] = all_sample_ids #json.dumps(all_sample_ids_)
    session['chosen'] = chosen_exp
    session['piece_name'] = piece_name_
    piece = pieces_dic[piece_name]
    # NOTE: we always sort the whole list

    sorted_names = list(sorted(piece.keys()))
    session['sorted_names'] = sorted_names
    # TODO: here we use both the sorted names (for deterministically choosing between experiemtns. We could change everywhere this to use ids.)
    chosen_id = sorted_names[chosen_exp]
    chosen_curve = piece[chosen_id]
    all_curves = [piece[curve_name] for curve_name in sorted_names]
    
    chosen_sample_folder = chosen_curve.joinpath(str(all_sample_ids[sorted_names.index(chosen_id)]))
    all_sample_folders = [p.joinpath(str(i)) for i, p in zip(all_sample_ids, all_curves)]
    chosen_plot_folder = chosen_sample_folder.joinpath('plotly_figs')
    
    session['piano_roll_gen'] = str(chosen_plot_folder.joinpath("piano_roll_1.json"))
    session['used_template'] = str(chosen_plot_folder.joinpath("ic_int_summed_1.json"))
    session['audio_file'] = str(chosen_sample_folder.joinpath("song.mp3"))
    session['templates'] = [str(p.joinpath('plotly_figs', "match_ic_int_summed.json")) for p in all_sample_folders]
    
piece_name = 'kv331_1_start_-45_nodes_134_end_0'
chosen = 0 # '0e3ee2b957cff8903c6c1eb79457a6add2cf5c2c8e88c94855ec5f6faa0b4f35'
# chosen = 1 # '3bceac27346acddd96b11d908f3ac92fd667561f0f2fad7622b5b3d5a4b6d47e'
# chosen = 2 # '80f8da596bf413a11494bedba1c1793778e648bda3bc2d2c1a49b7e2cca22669'
all_sample_ids = [str(0) for _ in range(len(configs))]


@app.route("/") 
def home():
    if state == 'DEBUG':
        render_page = "debug.html"
    elif state == 'RANDOM_SAMPLES':
        chosen = np.random.randint(0, len(configs))
        # TODO generalize
        all_sample_ids = np.random.randint(0, 8, len(configs)).tolist()
        piece_name_ = piece_name
        render_page = "random_samples.html"
    else:
        render_page = "main.html"
    set_state(chosen, piece_name_, all_sample_ids)
    audio_file = Path(session["audio_file"])
    audio_file = hashlib.sha256(str(audio_file.parent).encode('utf-8')).hexdigest() + ".mp3"
    return render_template(render_page, n_curves=len(session["templates"]), audio_path =url_for('audio', audio_file=audio_file))
 

@app.route("/curve/<int:curve_id>")
def curve(curve_id: int):
    curve = Path(session["templates"][curve_id])
    return send_file(curve.absolute(), as_attachment=False, max_age=0)
@app.route("/choose", methods=["POST"]) 
def choose():
    choice = request.form.get('choice', "")
    success = False
    with app.app_context():
        if choice and choice.isdigit():
            chosen = session["chosen"]
            correct_exp = session["sorted_names"][chosen]
            all_sample_ids = session["all_sample_ids"]
            correct_sample = all_sample_ids[chosen]
            choice = int(choice)
            selected_exp = session["sorted_names"][choice]
            selected_sample = all_sample_ids[choice]
            res = query_db("INSERT INTO result (piece, correct_exp, correct_sample, selected_exp, selected_sample ) VALUES (?,?,?,?,?)", (piece_name, correct_exp,correct_sample,selected_exp,selected_sample))
            if chosen == choice:
                success = True
    with open(session["used_template"], 'r') as f:
        template = json.load(f)
    return make_response(jsonify({"success": success, "correct": chosen, "generated_curve": template}), 200)
@app.route("/audio")
def audio():
    audio_file = Path(session["audio_file"])
    new_name = hashlib.sha256(str(audio_file.parent).encode('utf-8')).hexdigest() + ".mp3"
    response = send_file(
        audio_file.absolute(),
        # as_attachment=False,
        as_attachment=True,
        download_name=new_name,
        # last_modified=datetime.datetime.now()
        # max-age=,
        )
    # response.headers['Cache-Control'] = 'max-age=0, no-cache, must-revalidate, proxy-revalidate'
    return response
@app.route("/piano_roll")
def piano_roll():
    return send_file(Path(session["piano_roll_gen"]).absolute(), as_attachment=False, max_age=0)
@app.route("/template")
def template():
    if state == "DEBUG":
        return send_file(Path(session["used_template"]).absolute(), as_attachment=False, max_age=0)
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
    return redirect(url_for('home'))
if __name__ == "__main__": 
    app.run(debug=False) 