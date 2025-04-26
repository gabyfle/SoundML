"""
This file is part of SoundML.

Copyright (C) 2025 Gabriel Santamaria

This script is used to generate test vectors for SoundML.
The reference implementation choosen is librosa.
It's supposed to be ran only once. Then the generated vectors
haved to be used for the actual testing.
"""

from typing import Any, Tuple, Dict
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
            os.makedirs(self.output_dir)

        for audio_path in self.audio_paths:
            identifier = os.path.splitext(os.path.basename(audio_path))[0]
            print(
                f"Generating {self.BASE_IDENTIFIER} vector for: {identifier} (Audio: {audio_path})"
            )
            try:
                data: Tuple[np.ndarray, Parameters] = self.vector(audio_path)
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

    counter: int = 0

    def vector(self, audio_path: str) -> Tuple[np.ndarray, Parameters]:
        """
        Generate the time-series vector for the given file
        """
        mono = True if self.counter % 2 == 0 or self.counter % 3 == 0 else False
        sr = 22050 if (self.counter) % 2 == 0 and self.counter % 3 != 0 else None
        # For the moment, SoundML don't support soxr resampling, so we're using
        # samplerate's linear resampling
        y, sr = librosa.load(audio_path, mono=mono, sr=sr, res_type="linear")
        y = y.astype(np.float32)
        self.counter += 1
        return (y, Parameters({"sr": sr, "mono": mono}))


class STFTVectorGenerator(VectorGenerator):
    """
    Reads an audio file and creates a STFT vector representation of it
    """

    BASE_IDENTIFIER: str = "stft"

    def vector(self, audio_path: str) -> Tuple[np.ndarray, Parameters]:
        """
        Generate the STFT vector for the given file
        """
        y, sr = librosa.load(audio_path, sr=None)
        y = y.astype(np.float64)
        stft = librosa.stft(y)
        params = Parameters({"sr": sr, "n_fft": 2048, "hop_length": 512})
        return (stft, params)


generators: list[VectorGenerator] = [TimeSeriesVectorGenerator]

if __name__ == "__main__":
    audio_files = [
        os.path.join(AUDIO_DIRECTORY, f) for f in os.listdir(AUDIO_DIRECTORY)
    ]
    if not os.path.exists(VECTOR_DIRECTORY):
        os.makedirs(VECTOR_DIRECTORY)

    for generator in generators:
        generator = generator(audio_files, VECTOR_DIRECTORY)
        generator.generate()
