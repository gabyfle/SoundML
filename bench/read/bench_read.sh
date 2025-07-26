#!/bin/bash

OCAML_CMD="dune exec ./bench/read/perf.exe"
PYTHON_CMD="python3 bench.py"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CACHE_CLEAR_CMD="sync; echo 3 > /proc/sys/vm/drop_caches"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    CACHE_CLEAR_CMD="purge"
else
    CACHE_CLEAR_CMD=""
fi

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <mode> <num_iterations> <root_directory> <sample_rate> <format> <max_files>"
    echo "  <mode>: 'ocaml' or 'python'"
    exit 1
fi

MODE=$1
shift

NUM_ITERATIONS=$1
ROOT_DIR=$2
SAMPLE_RATE=$3
FORMAT=$4
MAX_FILES=$5

if [[ "$MODE" != "ocaml" && "$MODE" != "python" ]]; then
    echo "Error: <mode> must be either 'ocaml' or 'python'. You provided '$MODE'."
    echo "Usage: $0 <mode> <num_iterations> <root_directory> <sample_rate> <format> <max_files>"
    exit 1
fi

if ! [[ "$NUM_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
     echo "Error: <num_iterations> ('$NUM_ITERATIONS') must be a positive integer." >&2
     exit 1
fi

BENCH_CMD=""
if [[ "$MODE" == "ocaml" ]]; then
    BENCH_CMD="$OCAML_CMD"
elif [[ "$MODE" == "python" ]]; then
    BENCH_CMD="$PYTHON_CMD"
fi

results_array=()
valid_run_count=0

echo "Starting reading test for: $MODE"
echo "Command to run: $BENCH_CMD \"$ROOT_DIR\" \"$SAMPLE_RATE\" \"$FORMAT\" \"$MAX_FILES\""
echo "Number of iterations: $NUM_ITERATIONS"
echo "OS detected: $OSTYPE"
echo "--------------------------------------------------"

for (( i=1; i<=NUM_ITERATIONS; i++ )); do
    echo "Iteration $i / $NUM_ITERATIONS"
    
    # Clear cache based on OS
    if [[ -n "$CACHE_CLEAR_CMD" ]]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            if sudo bash -c "$CACHE_CLEAR_CMD"; then
                sleep 1.5
            else
                echo "Warning: Failed to clear cache on Linux. Continuing without cache clearing." >&2
            fi
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if sudo $CACHE_CLEAR_CMD; then
                sleep 1.5
            else
                echo "Warning: Failed to clear cache on macOS. Continuing without cache clearing." >&2
            fi
        fi
    else
        echo "Warning: Cache clearing not supported on this OS. Continuing without cache clearing." >&2
    fi
    
    result=$( $BENCH_CMD "$ROOT_DIR" "$SAMPLE_RATE" "$FORMAT" "$MAX_FILES" )
    exit_status=$?

    if [ $exit_status -ne 0 ]; then
        continue
    fi

    if [[ "$result" =~ ^[+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$ ]]; then
        echo "Result: $result MiB/s"
        results_array+=("$result")
        ((valid_run_count++))
    fi
    sleep 1
done

echo "--------------------------------------------------"

num_results=${#results_array[@]}

if [ "$num_results" -eq 0 ]; then
    exit 1
fi
stats=$(printf "%s\n" "${results_array[@]}" | awk '
    NF == 0 { next }
    {
        if ($1 ~ /^[+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$/) {
            sum += $1;
            sumsq += $1*$1;
            count++;
        }
    }
    END {
        if (count > 0) {
            mean = sum / count;
            if (count > 1) {
               variance = (sumsq - (sum*sum)/count) / (count-1);
               if (variance < 1e-12) variance = 0;
               stdev = sqrt(variance);
            } else {
               stdev = 0; # Standard deviation is undefined/0 for a single point
            }
            printf "%.6f %.6f %d", mean, stdev, count;
        } else {
            print "NaN NaN 0";
        }
    }
')
read -r mean stdev count <<< "$stats"

if [[ -z "$mean" || -z "$stdev" || -z "$count" || "$count" -eq 0 ]]; then
    echo "Error: Failed to calculate statistics. Awk output: '$stats'" >&2
    exit 1
fi

echo "Performance Test Summary ($MODE):"
echo "-------------------------"
echo "Command: $BENCH_CMD \"$ROOT_DIR\" \"$SAMPLE_RATE\" \"$FORMAT\" \"$MAX_FILES\""
printf "Mean Speed (MiB/s): %.6f\n" "$mean"
if [ "$count" -gt 1 ]; then
    printf "Std Dev Speed:        %.6f\n" "$stdev"
else
    printf "Std Dev Speed:        N/A (requires >= 2 data points)\n"
fi

exit 0
