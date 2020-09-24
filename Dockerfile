# Definition of Submission container
ARG AIDO_REGISTRY



FROM ${AIDO_REGISTRY}/duckietown/challenge-aido_lf-template-pytorch:daffy


# let's create our workspace, we don't want to clutter the container
RUN rm -r /workspace; mkdir /workspace

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN echo PIP_INDEX_URL=${PIP_INDEX_URL}


# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
COPY requirements.* ./
RUN pip install -U pip>=20.2
RUN pip install --use-feature=2020-resolver -r requirements.resolved

# let's copy all our solution files to our workspace
# if you have more file use the COPY command to move them to the workspace
COPY solution.py /workspace
COPY models /workspace/models
COPY model.py /workspace
COPY wrappers.py /workspace

# we make the workspace our working directory
WORKDIR /workspace


# let's see what you've got there...
CMD python solution.py
