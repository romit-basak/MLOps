# Docker Lab 2: train and serve an Iris classifier with a multi-stage build

This lab packages a small TensorFlow model in Docker. The first build stage trains a Keras classifier on the Iris dataset, and the second stage copies the trained artifacts into a fresh image that serves predictions through a Flask web app. A `docker-compose.yml` file runs the same two steps as two separate containers that share a volume.

My changes target the handoff between the two stages. The original lab built and started, but the server could not be reached on its published port and returned "Virginica" for every flower. I fixed both problems and added an accuracy check to the training stage that stops the build when the model is not good enough.

## Summary of changes

| | Original lab | After my changes |
|---|---|---|
| Port | Image exposes 80, compose maps `80:80`, Flask listens on 4000, so the documented port returns an empty reply | Flask, `EXPOSE`, compose and `HOWTO` all use port 4000 |
| Artifacts copied from stage 1 to stage 2 | `my_model.keras` only | `my_model.keras`, `scaler.joblib` and `metrics.json` |
| Predictions on 5 known flowers | 2 of 5 correct (every input returned "Virginica") | 5 of 5 correct |
| Model quality check | None; any model became an image | Build fails if test accuracy is below `MIN_ACCURACY` (default 0.90) |
| Test accuracy | 0.8667 (50 epochs, never checked) | 0.9667 (200 epochs, fixed seed) |
| Editing the training script | Re-ran the full `pip install` | Reuses the cached pip layer |

The details of each change are further down, under [What I changed and why](#what-i-changed-and-why).

## Requirements

You only need Docker. Python, TensorFlow and the other libraries are installed inside the containers.

| Tool | Version I tested with | Notes |
|---|---|---|
| Docker Engine | 29.5.2 | On macOS or Windows, install [Docker Desktop](https://www.docker.com/products/docker-desktop/). On Linux, install [Docker Engine](https://docs.docker.com/engine/install/). |
| Docker Compose | v5.6.0 | Included with Docker Desktop. On Linux, install the `docker-compose-plugin` package. Only needed for the Compose section. |
| Git | any recent version | To clone the repository. You can also download it as a ZIP from GitHub. |
| curl | any | Optional, for testing from the terminal. A browser works too. |

Other things to know before you start:

- Each image is about 4.2 GB, because the base image is the full `python:3.10` and TensorFlow 2.15 is large. Leave at least 10 GB of free disk space.
- The first build downloads the base image and Python packages and took about 3.5 minutes on my machine. Later builds reuse the cache.
- I tested on macOS with Apple Silicon (arm64) using Docker through Colima. TensorFlow 2.15 also publishes wheels for x86_64 Linux, so Intel machines and Windows with WSL 2 should work the same way.
- Port 4000 on your machine must be free. See [Troubleshooting](#troubleshooting) if it is not.

Check that Docker is running:

```bash
docker version
docker compose version
```

Both commands should print a version. If `docker version` shows a client but reports that it cannot connect to the daemon, start Docker Desktop (or `colima start` if you use Colima) and try again.

## Get the code

```bash
git clone https://github.com/romit-basak/MLOps.git
cd MLOps/Labs/Docker_Labs/Lab2
```

Run every command below from this `Lab2` folder.

## Option 1: build and run the image

Build the image. This trains the model inside the build:

```bash
docker build -t app .
```

Near the end of the training step, the output should include these lines:

```
Test accuracy: 0.9667 (minimum required: 0.9000)
Model was trained and saved
```

The build finishes with `Successfully tagged app:latest` (older builder) or `naming to docker.io/library/app:latest` (BuildKit). With BuildKit, the training output is collapsed by default; add `--progress=plain` to see it.

Start the server:

```bash
docker run --rm -p 4000:4000 app
```

The container log should show:

```
Loaded model with test accuracy 0.9667
 * Running on http://127.0.0.1:4000
```

Leave this terminal open. Press Ctrl+C to stop the server when you are done.

### What a successful run looks like

Open http://localhost:4000/predict in a browser, type in four measurements and click Analyze. The predicted species appears with a picture of the flower. Try these values, taken from the Iris dataset:

| Sepal length | Sepal width | Petal length | Petal width | Expected |
|---|---|---|---|---|
| 5.1 | 3.5 | 1.4 | 0.2 | Setosa |
| 5.9 | 3.0 | 4.2 | 1.5 | Versicolor |
| 6.7 | 3.0 | 5.2 | 2.3 | Virginica |

You can also test from a second terminal:

```bash
curl http://localhost:4000/
# Welcome to the Iris Classifier API!

curl -X POST http://localhost:4000/predict \
  -d sepal_length=5.1 -d sepal_width=3.5 -d petal_length=1.4 -d petal_width=0.2
```

Expected response:

```json
{
  "predicted_class": "Setosa"
}
```

The Virginica sample (`6.7`, `3.0`, `5.2`, `2.3`) should return `"Virginica"`.

## Option 2: run with Docker Compose

Compose runs training and serving as two containers. Training writes the model, scaler and metrics to a named volume, and serving starts only after training exits successfully.

```bash
docker compose up
```

The first run installs the Python packages in both containers, so it takes a few minutes. When it is ready, the log shows:

```
ml_trainer  | Test accuracy: 0.9667 (minimum required: 0.9000)
ml_trainer  | Model was trained and saved
ml_serving  | Loaded model with test accuracy 0.9667
ml_serving  |  * Running on http://127.0.0.1:4000
```

Test it in the browser or with the same `curl` commands as above. Stop it with Ctrl+C, then remove the containers and the volume:

```bash
docker compose down -v
```

## Try the accuracy check

The training script compares test accuracy with `MIN_ACCURACY` and exits with code 1 if the model falls short. Set the minimum above what the model can reach to watch it refuse:

```bash
docker build --build-arg MIN_ACCURACY=0.99 -t app-strict .
```

Expected output:

```
Test accuracy: 0.9667 (minimum required: 0.9900)
Accuracy gate FAILED: model is below the minimum, aborting the build
The command '/bin/sh -c MIN_ACCURACY=${MIN_ACCURACY} python model_training.py' returned a non-zero code: 1
```

No `app-strict` image is created; `docker images app-strict` lists nothing.

The same check works with Compose:

```bash
MIN_ACCURACY=0.99 docker compose up
```

The trainer exits with code 1, Compose prints `service "model-training" didn't complete successfully: exit 1`, and the `ml_serving` container is created but never started, so nothing answers on port 4000. Run `docker compose down -v` afterwards.

## What I changed and why

### 1. Fixed the port mismatch

Files: `dockerfile`, `docker-compose.yml`, `src/main.py`, `HOWTO`

The original Dockerfile had `EXPOSE 80` and the compose file mapped `80:80`, but `main.py` started Flask on port 4000. Nothing inside the container listened on port 80, so a request to the documented port got an empty reply (curl exit code 52). `HOWTO` also ran `docker run app` without `-p`, which publishes no ports.

Flask now reads its port from a `PORT` environment variable that defaults to 4000. The Dockerfile sets `ENV PORT=4000` and `EXPOSE 4000`, compose maps `"4000:4000"`, and `HOWTO` uses `docker run -p 4000:4000 app`.

### 2. Copied the fitted scaler from the training stage to the serving stage

Files: `src/model_training.py`, `src/main.py`, `dockerfile`, `docker-compose.yml`

Training standardizes the four features with scikit-learn's `StandardScaler`, so the model learns from values centered on 0. Only `my_model.keras` was copied into the serving stage, and the server passed raw measurements in centimetres straight to the model. Values like 5.1 or 6.7 are far outside what it was trained on, and on the original image it answered "Virginica" for all five test flowers.

The training stage now saves the fitted scaler as `scaler.joblib`, and the serving stage copies it alongside the model:

```dockerfile
COPY --from=model_training /app/my_model.keras /app/scaler.joblib /app/metrics.json /app/
```

`main.py` loads the scaler and calls `scaler.transform()` on each request before predicting. In compose, the trainer copies all three files into the shared `model_exchange` volume and the server copies them out. With this change, all five test flowers are classified correctly.

### 3. Added an accuracy check to the training stage

Files: `src/model_training.py`, `dockerfile`, `docker-compose.yml`

After training, the script evaluates the model on the 30 held-out test flowers and exits with code 1 if accuracy is below `MIN_ACCURACY`. In a Dockerfile, a `RUN` step that exits non-zero stops the build, so a model below the bar never reaches a serving image. The threshold is a build argument, `ARG MIN_ACCURACY=0.9`, and can be changed with `--build-arg`. In compose, it is an environment variable with the same default, and `depends_on: condition: service_completed_successfully` keeps the server from starting after a failed run.

The script also writes `metrics.json` with the test accuracy and the threshold used. That file is copied into the serving image, and the server prints the accuracy when it starts, so you can tell which model a running container holds.

`tf.keras.utils.set_random_seed(42)` fixes the weight initialization, so the same code gives the same accuracy and the same pass or fail result on every build.

### 4. Trained for 200 epochs instead of 50

File: `src/model_training.py`

The new check failed on the lab's own model. With 50 epochs, test accuracy was 0.8667, below the 0.90 minimum. I trained the same network for different numbers of epochs inside the training container:

| Epochs | Test accuracy |
|---|---|
| 50 | 0.8667 |
| 100 | 0.9000 |
| 150 | 0.9333 |
| 200 | 0.9667 |

At 100 epochs the model only just meets the minimum, so I chose 200, which passes with some room. The network itself is unchanged.

### 5. Reordered the training stage for layer caching

File: `dockerfile`

Docker reuses a cached layer only while every earlier step is unchanged. The original copied `model_training.py` before `requirements.txt`, so any edit to the training script forced a full `pip install` of TensorFlow again. The stage now copies `requirements.txt` and installs packages first, then copies the script. A rebuild after editing the script or changing `MIN_ACCURACY` reuses the install layer and only re-runs training.

## Files

| File | Purpose |
|---|---|
| `dockerfile` | Two-stage build: `model_training` trains and checks the model, `serving` runs the Flask app with the copied artifacts |
| `docker-compose.yml` | Runs training and serving as two containers that share the `model_exchange` volume |
| `requirements.txt` | Python packages (TensorFlow 2.15.0, scikit-learn, Keras, Flask, requests) |
| `src/model_training.py` | Trains the model, runs the accuracy check, saves `my_model.keras`, `scaler.joblib` and `metrics.json` |
| `src/main.py` | Flask app: `/` returns a welcome message, `GET /predict` serves the form, `POST /predict` returns a prediction |
| `src/templates/predict.html` | Web form |
| `src/statics/` | Flower images shown with each prediction |
| `HOWTO` | The two commands for building and running the image |

## Troubleshooting

`Bind for 0.0.0.0:4000 failed: port is already allocated`: something else is using port 4000. Publish the container on another host port, for example `docker run --rm -p 5000:4000 app`, and open http://localhost:5000/predict. For compose, change the left side of the mapping to `"5000:4000"` in `docker-compose.yml`.

`Cannot connect to the Docker daemon`: Docker is installed but not running. Start Docker Desktop or run `colima start`.

The browser shows "SYSTEM ERROR: Neural Link Disconnected": the server returned an error. Check the terminal running the container for a Python traceback, and make sure all four fields contain numbers.

The build stops with `Accuracy gate FAILED`: the model did not reach `MIN_ACCURACY`. With the default of 0.9 this should not happen, because the seed is fixed. If you changed the training code or passed a higher `--build-arg`, that is the check working as intended.

## Cleanup

```bash
docker rmi app app-strict 2>/dev/null
docker compose down -v
docker image prune
```

`docker image prune` removes the untagged intermediate images left behind by the training stage. It asks for confirmation first.
