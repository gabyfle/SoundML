"""
This file is part of SoundML.

Copyright (C) 2025 Gabriel Santamaria

This script is used to generate test vectors for SoundML.
The reference implementation choosen is librosa.
It's supposed to be ran only once. Then the generated vectors
haved to be used for the actual testing.
"""

from typing import Any, Tuple, Dict, List
import os
import json
import numpy as np
import librosa

AUDIO_DIRECTORY = "audio/"
VECTOR_DIRECTORY = "vectors/"


class Parameters:
    """
    Class representing parameters used to generate a vector
    """

    parameters: Dict[str, Any]

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    def write(self, filename: str):
        """
        Write the parameters to a JSON file
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.parameters, f, indent=4)


class VectorGenerator:
    """
    Abstract class representing an audio vector generators
    """

    BASE_IDENTIFIER: str

    audio_paths: list[str]
    output_dir: str

    counter: int = 0

    def __init__(self, audio_paths: list[str], output_dir: str):
        self.audio_paths = audio_paths
        self.output_dir = os.path.join(output_dir, f"{self.BASE_IDENTIFIER}/")

    def normalize_name(self, name: str) -> str:
        """
        Normalize the name of the audio file
        """
        return name.replace(" ", "_").replace("-", "_").lower()

    def vector(self, audio_path: str) -> Tuple[np.ndarray, Parameters]:
        """
        Generate the vector for the given audio file
        """
        raise NotImplementedError("Subclasses should implement this method")

    def generate(self):
        """
        Generate the audio vectors
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        for audio_path in self.audio_paths:
            identifier = os.path.splitext(os.path.basename(audio_path))[0]
            try:
                data: Tuple[np.ndarray, Parameters] = self.vector(audio_path)

                self.counter += 1

                y: np.ndarray = data[0]
                params = data[1]
                filename: str = self.normalize_name(
                    f"{self.BASE_IDENTIFIER}_{identifier}"
                )
                output_filename: str = os.path.join(self.output_dir, f"{filename}.npy")
                params_filename: str = os.path.join(self.output_dir, f"{filename}.json")
                np.save(output_filename, y)
                params.write(params_filename)
            except Exception as e:
                print(f"ERROR generating for {identifier}: {e}")


class TimeSeriesVectorGenerator(VectorGenerator):
    """
    Reads an audio file and creates a time-series vector representation of it
    """

    BASE_IDENTIFIER: str = "timeseries"

    resamplers: List[str] = ["soxr_vhq", "soxr_hq", "soxr_mq", "soxr_lq"]
    srs = [None, 8000, 16000, 22050]

    def vector(self, audio_path: str) -> Tuple[np.ndarray, Parameters]:
        """
        Generate the time-series vector for the given file
        """
        params = {}
        mono = False if self.counter % 2 == 0 or self.counter % 3 == 0 else False
        sr = self.srs[self.counter % len(self.srs)]
        res_type = self.resamplers[self.counter % len(self.resamplers)]
        params["mono"] = mono
        if sr is not None:
            params["res_type"] = res_type
        y, sr = librosa.load(
            audio_path, mono=mono, sr=sr, res_type=res_type, dtype=np.float64
        )
        params["sr"] = sr
        y = np.ascontiguousarray(y, dtype=np.float64)

        return (y, Parameters(params))


class STFTVectorGenerator(VectorGenerator):
    """
    Reads an audio file and creates a STFT vector representation of it
    """

    BASE_IDENTIFIER: str = "stft"

    nffts = [512]#, #1024, 2048, 4096]
    window_lengths = [512]#64, 128, 256, 512]
    hop_sizes = [128]#, 256, 512]
    centers = [False, False, False]
    window_types = ["hann"]#, "hamming", "blackman", "boxcar"]

    def vector(self, audio_path: str) -> Tuple[np.ndarray, Parameters]:
        """
        Generate the STFT vector for the given file
        """
        params = {}
        n_fft = self.nffts[self.counter % len(self.nffts)]
        hop_size = self.hop_sizes[self.counter % len(self.hop_sizes)]
        window_type = self.window_types[self.counter % len(self.window_types)]
        window_length = self.window_lengths[self.counter % len(self.window_lengths)]
        center = self.centers[self.counter % len(self.centers)]
        params["window_length"] = window_length
        params["n_fft"] = n_fft
        params["hop_size"] = hop_size
        params["window_type"] = window_type
        params["center"] = center
        params["res_type"] = "soxr_hq"

        y, sr = librosa.load(audio_path)
        y = y.astype(np.float64)
        stft = librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=window_length,
            window=window_type,
            dtype=np.complex64,
            center=center,
        )
        stft = np.ascontiguousarray(stft, dtype=np.complex64)
        params = Parameters(params)

        return (stft, params)


generators: list[VectorGenerator] = [TimeSeriesVectorGenerator, STFTVectorGenerator]

if __name__ == "__main__":
    audio_files = [
        os.path.join(AUDIO_DIRECTORY, f) for f in os.listdir(AUDIO_DIRECTORY)
    ]
    if not os.path.exists(VECTOR_DIRECTORY):
        os.makedirs(VECTOR_DIRECTORY)

    for generator in generators:
        generator: VectorGenerator = generator(audio_files, VECTOR_DIRECTORY)
        generator.generate()
