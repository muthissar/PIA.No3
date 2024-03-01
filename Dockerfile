# FROM python:3.9.2-alpine
# FROM python:3.8-alpine
# ARG PIA_DIR='../../PIA.No3'
# ARG PIA_DIR='/home/mathias/devel/python/PIA.No3'
FROM continuumio/miniconda3:23.5.2-0-alpine

# upgrade pip
# RUN pip install --upgrade pip

# get curl for healthchecks
RUN apk add curl

# permissions and nonroot user for tightened security
RUN adduser -D nonroot
RUN mkdir /home/app/ && chown -R nonroot:nonroot /home/app
RUN mkdir -p /var/log/flask-app && touch /var/log/flask-app/flask-app.err.log && touch /var/log/flask-app/flask-app.out.log
RUN chown -R nonroot:nonroot /var/log/flask-app
WORKDIR /home/app
USER nonroot

# # copy all the files to the container
# COPY --chown=nonroot:nonroot . .

# # venv
# ENV VIRTUAL_ENV=/home/app/venv

# # python setup
# RUN python -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# RUN export FLASK_APP=app.py
# RUN pip install -r requirements.txt

# # define the port number the container should expose
# EXPOSE 5000

# CMD ["python", "app.py"]

# COPY --chown=nonroot:nonroot . .
# TODO change
COPY --chown=nonroot:nonroot . .
RUN conda env create -n pia-web -f environment.yml
# RUN conda install -n base conda-libmamba-solver
# RUN conda config --set solver libmamba
# ENV PATH /opt/conda/envs/mro_env/bin:$PATH
# RUN /bin/bash -c "source activate mro_env"
RUN echo "source activate pia-web" > ~/.bashrc
ENV PATH /home/nonroot/.conda/envs/pia-web/bin:$PATH
RUN pip install "./jazz-music21"
RUN pip install "./DatasetManager"


EXPOSE 5000
# The code to run when container is started:
CMD ["python", "app.py"]
