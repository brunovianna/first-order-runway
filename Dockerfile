FROM runwayml/model-runner:gpu-python3.6-cuda10.0
COPY . /model
WORKDIR /model
RUN model_runner --build .
ENTRYPOINT []
CMD python runway_model.py
