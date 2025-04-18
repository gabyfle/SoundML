"""
This file is part of SoundML.

Copyright (C) 2025 Gabriel Santamaria

This script is used to generate test vectors for SoundML.
The reference implementation choosen is librosa.
It's supposed to be ran only once. Then the generated vectors
haved to be used for the actual testing.
"""

import os
import numpy as np
import librosa

AUDIO_DIRECTORY = "audio/"
VECTOR_DIRECTORY = "test_vectors/"


class VectorGenerator:
    """
    Abstract class representing an audio vector generators
    """

    BASE_IDENTIFIER: str

    audio_paths: list[str]
    output_dir: str

    def __init__(self, audio_paths: list[str], output_dir: str):
        self.audio_paths = audio_paths
        self.output_dir = output_dir

    def normalize_name(self, name: str) -> str:
        """
        Normalize the name of the audio file
        """
        return name.replace(" ", "_").replace("-", "_").lower()

    def vector(self, audio_path: str) -> np.ndarray:
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
                y = self.vector(audio_path)
                filename: str = self.normalize_name(
                    f"{self.BASE_IDENTIFIER}_{identifier}"
                )
                output_filename: str = os.path.join(self.output_dir, f"{filename}.npy")
                np.save(output_filename, y)
            except Exception as e:
                print(f"ERROR generating for {identifier}: {e}")


class TimeSeriesVectorGenerator(VectorGenerator):
    """
    Reads an audio file and creates a time-series vector representation of it
    """

    BASE_IDENTIFIER: str = "timeseries"

    def vector(self, audio_path: str) -> np.ndarray:
        """
        Generate the time-series vector for the given file
        """
        y, _ = librosa.load(audio_path, sr=None)
        y = y.astype(np.float64)
        return y


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
