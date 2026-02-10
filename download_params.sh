#!/bin/bash
set -euo pipefail
mkdir -p params
curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C params
echo "AF2 parameters downloaded to params/"
