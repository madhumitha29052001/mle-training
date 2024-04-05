#!/bin/bash

conda run -n housing python -m housing.ingest_data
conda run -n housing python -m housing.train 
conda run -n housing python -m housing.score