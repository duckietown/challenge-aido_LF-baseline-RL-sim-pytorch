# Definition of Submission container
ARG AIDO_REGISTRY



FROM ${AIDO_REGISTRY}/duckietown/challenge-aido_lf-template-pytorch:daffy-amd64


# let's create our workspace, we don't want to clutter the container
RUN rm -r /workspace; mkdir /workspace


# we make the workspace our working directory
WORKDIR /workspace

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN echo PIP_INDEX_URL=${PIP_INDEX_URL}


# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
RUN pip install -U "pip>=20.2"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN  pip3 install --use-feature=2020-resolver -r .requirements.txt

# Juuuuust in case...
RUN pip3 uninstall dataclasses -y

# let's copy all our solution files to our workspace
# if you have more file use the COPY command to move them to the workspace
COPY *.py ./
COPY models /workspace/models




CMD python3 solution.py
