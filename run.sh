#!/usr/bin/env bash
set -e

python -m src.data_pipeline
python -m src.train
python -m src.eval
python -m src.nlp_classifier
python -m src.rl_agent