#!/usr/bin/env bash
set -e
python -m src.generate_dataset
python -m src.train
python -m src.evaluate
