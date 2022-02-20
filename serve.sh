#!/bin/bash
source .env/bin/activate
mlflow models serve -m mlruns/0/runid/artifacts/model