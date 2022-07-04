#!/bin/bash

cd preprocess
../venv/bin/python create_subgroups.py
../venv/bin/python embeddings_dataset.py
../venv/bin/python verify_dataset.py
../venv/bin/python answer_starters.py

cd ..

cd xgboost
../venv/bin/python train.py
