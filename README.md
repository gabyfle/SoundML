<div align='center'>

<img src="soundml_logo.svg" width="140px" alt="SoundML Logo">

<h1>SoundML</h1>
<p>A little and very high level library to perform basic operations on audio files in the OCaml language</p>

<h4> <span> · </span> <a href="https://github.com/gabyfle/SoundML/blob/master/README.md"> Documentation </a> <span> · </span> <a href="https://github.com/gabyfle/SoundML/issues"> Report Bug </a> <span> · </span> <a href="https://github.com/gabyfle/SoundML/issues"> Request Feature </a> · </h4>


</div>

## About the Project

> [!WARNING]
> The project is still in development and is not yet ready for use.

## Getting Started

### Installation

This project uses Opam as a package manager
```bash
opam install soundml
```

## Roadmap

* [x] Read and Write audio
* [x] Compute the FFT of an audio signal
* [x] Compute the IFFT of an FFT
* [ ] Compute the spectrogram of an audio file
  * [ ] Use a generic spectral function (similar to `matplotlib.mlab`) to compute the spectrogram
  * [ ] Allow different types of specgram computation (mel, mag, etc...)
* [x] Audio slicing (in a similar way to Owl's slicing)
* [ ] Output spectrograms and ffts as an image (PNG) (using FFMPEG's library to output a PNG out of Owl's NDarray)
* [ ] Basic onset detection algorithms
* [ ] RMS
  * [ ] LUFS and DBSF are left for further version (LUFS algorithm might require in-depth review of the standards) 

## Requirements

You should be using the OCaml compiler with a version at least equal to 5.1.0. You can install it by following the instructions on the [OCaml website](https://ocaml.org/docs/install.html). This project uses the Dune build system.

This library heavily relies on the Owl and ocaml-ffmpeg libraries.

<div align=center>

| Name                                                                                                  | Version     | Description                                                                                        |
| ----------------------------------------------------------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------- |
| [**Owl**](https://github.com/owlbarn/owl) - *OCaml Scientific Computing*                              | `>= 1.1`    | Library for scientific computing in OCaml. Used to make the heavy computations (FFT, IFFT, etc...) |
| [**ocaml-ffmpeg**](https://github.com/savonet/ocaml-ffmpeg) - *OCaml bindings to the FFmpeg library.* | `>= 1.1.11` | OCaml bindings for FFmpeg. Used to read and write audio data.                                      |

</div>

## Inspirations

This project is heavily inspired by other amazing open-source libraries such as:


<div align=center>

| Name                                              | Inspiration                                                    | Reference                                                                                                                                                                                                                                 |
| ------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [**librosa**](https://github.com/librosa/librosa) | General functionalities for audio signal processing            | McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. "librosa: Audio and music signal analysis in python." In Proceedings of the 14th python in science conference, pp. 18-25. 2015. |
| [**pydub**](https://github.com/jiaaro/pydub)      | Ease of use, audio slicing using milliseconds and manipulation | -                                                                                                                                                                                                                                         |

</div>

Don't hesitate to check the amazing work done by the authors and contributors of these libraries!

## License

Distributed under the Apache License Version 2.0. See LICENSE for more information.

## Acknowledgements

* Logo generated with DALL-E by OpenAI
