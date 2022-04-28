#!/bin/bash
A=("$@")
B=("${A[0]}")
C=("${A[@]:1}")

echo "Importing conda functions..."
source $(conda info --base)/etc/profile.d/conda.sh
echo "Activating env "$B"..."
conda activate "$B"
echo "Running "$C" in env "$B"..."
"$C"
