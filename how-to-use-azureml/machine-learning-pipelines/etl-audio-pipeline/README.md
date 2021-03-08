# Extract, Transform, Load Audio Azure Machine Learning Pipeline

## Installation

To install project dependencies run

```bash
pip install -r requirements.txt
```

Alternatively, if you are planning on contributing you will also need to obtain the development requirements

```bash
pip install -r dev-requirements.txt
```

To update `requirements.txt` run

```bash
pip-compile
```

To update `dev-requirements.txt` run

```bash
pip-compile dev-requirements.in
```

## Usage

To execute the pipeline script, from this directory run

```bash
python -m mlops.create_etl_pipeline
```
