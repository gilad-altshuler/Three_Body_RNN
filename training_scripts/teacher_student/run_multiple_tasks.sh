#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_PATH")")")"

# Run flip-flop jobs
echo "Starting flip-flop jobs..."
bash training_scripts/teacher_student/run_multiple_flipflop.sh

# After flip-flop jobs finish, run sine jobs
echo "Starting sine jobs..."
bash training_scripts/teacher_student/run_multiple_sin.sh

echo "All jobs completed."

# Now call collect data after all the jobs are finished
python "$ROOT/training_scripts/teacher_student/collect_data.py"
