#!/usr/bin/env python3

import os
import sys
import time
import argparse
import librosa
import numpy as np

MIB_DIVISOR = 1024.0 * 1024.0


def find_audio_files(root_dir, extension, max_files):
    filepaths = []
    extension = extension.lower()

    count = 0
    try:
        for dirpath, _, filenames in os.walk(root_dir, topdown=True, onerror=None):
            relevant_filenames = [f for f in filenames if f.lower().endswith(extension)]
            for filename in relevant_filenames:
                if count < max_files:
                    filepaths.append(os.path.join(dirpath, filename))
                    count += 1
                else:
                    return filepaths
            if count >= max_files:
                break

    except OSError:
        pass

    return filepaths


def get_file_size(filename) -> float:
    if not os.path.isfile(filename):
        return -1.0
    size = os.path.getsize(filename)
    if size <= 0:
        return 0.0
    return size / MIB_DIVISOR


def benchmark_read(filename, target_sr) -> tuple[float, float]:
    size = get_file_size(filename)
    if size is None or size <= 0.0:
        return 0.0, 0.0

    sample_rate = target_sr if target_sr is not None and target_sr > 0 else None

    try:
        start_time = time.perf_counter()
        _audio, _sr = librosa.load(
            filename, sr=sample_rate, mono=False, dtype=np.float32
        )
        end_time = time.perf_counter()
        duration = end_time - start_time
        if not isinstance(_audio, np.ndarray) or _audio.size == 0:
            return -1.0, -1.0

        return duration, size
    except FileNotFoundError:
        return -1.0, -1.0


def run_benchmark(root_dir, sample_rate, extension, max_files):
    all_files = find_audio_files(root_dir, extension, max_files)
    nfound = len(all_files)

    if nfound == 0:
        sys.exit(0)

    warmup_count = min(5, nfound)
    if warmup_count > 0:
        warmup_files = all_files[:warmup_count]
        for f in warmup_files:
            _ = benchmark_read(f, sample_rate)

    total_time = 0.0
    total_size = 0.0

    files_to_process = all_files

    for filename in files_to_process:
        duration, size = benchmark_read(filename, sample_rate)

        if duration > 0 and size > 0:
            total_time += duration
            total_size += size

    if total_time > 0 and total_size > 0:
        avg_speed = total_size / total_time
        print(f"{avg_speed:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_directory", help=argparse.SUPPRESS)
    parser.add_argument("sample_rate", type=int, help=argparse.SUPPRESS)
    parser.add_argument("format", help=argparse.SUPPRESS)
    parser.add_argument("max_files", type=int, help=argparse.SUPPRESS)

    if len(sys.argv) != 5:
        print(
            f"Usage: {sys.argv[0]} <root_directory> <sample_rate> <format> <max_files>",
            file=sys.stderr,
        )
        sys.exit(1)

    args = parser.parse_args()
    if not os.path.isdir(args.root_directory):
        sys.exit(1)

    if args.sample_rate < 0:
        sys.exit(1)

    if args.max_files <= 0:
        sys.exit(1)

    ext = args.format.lstrip(".")

    run_benchmark(args.root_directory, args.sample_rate, ext, args.max_files)


if __name__ == "__main__":
    main()
