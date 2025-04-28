## SoundML IO C++ library

This is the C/C++ code for SoundML IO library. It relies on C++23 (for `std::expected`).

### Dependencies

- `sndfile`: for reading and writing audio files (and *very* soon<span>&trade;</span> streams)
- `soxr`: for everything related to resampling

### General information

Since we're working with `Owl` that uses the `Bigarray.c_layout` layout, we choose to maintain the interleaved layout for the audio data. This allows us to directly write the data into the `Bigarray` without having to deinterleave it. Thus, when reading an audio files that contains `n` channels and `m` samples per channel, the final shape of the `Bigarray` will be `(m, n)` instead of `(n, m)` as you may be used to using other well known libraries (like *librosa*).

In this directory, you'll find the following files:

- `common.hxx` : contains the common functions and types used by both the reader and the writer.
- `read.hxx` : implements the needed `read`s functions. The file reading implementation is split between two classes:
  - `SoundML::IO::SndfileReader` : this is used when no resampling is needed.
  - `SoundML::IO::SoXrReader` : this is used when resampling is needed. It performs resampling while reading the file. Each read buffer is fed to the soxr resampler and the output is written directly to the `Bigarray` data pointer.

Exceptions are used to handle the errors. To do so, in `common.hxx` we retreive (inside `raise_caml_exception`) the correct exception to raise in OCaml based on the `Error` provided.
