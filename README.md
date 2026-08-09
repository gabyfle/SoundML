<div align='center'>

<img src="soundml_logo.svg" width="140px" alt="SoundML Logo">

<h1>SoundML</h1>
<p>A little and very high level library to perform basic operations on audio files in the OCaml language</p>

<h4> <span> · </span> <a href="https://github.com/gabyfle/SoundML/blob/master/README.md"> Documentation </a> <span> · </span> <a href="https://github.com/gabyfle/SoundML/issues"> Report Bug </a> <span> · </span> <a href="https://github.com/gabyfle/SoundML/issues"> Request Feature </a> · </h4>

[![Build](https://github.com/gabyfle/SoundML/actions/workflows/build.yml/badge.svg)](https://github.com/gabyfle/SoundML/actions/workflows/build.yml)
[![Test](https://github.com/gabyfle/SoundML/actions/workflows/test.yml/badge.svg)](https://github.com/gabyfle/SoundML/actions/workflows/test.yml)
</div>

## About the Project

> [!WARNING]
> The project is being rebuilt from the ground up and is not yet ready for use.
> The public API is being redesigned and nothing here is stable.

SoundML is distributed as two opam packages:

| Package | Contents |
| --- | --- |
| `soundml` | the core library; depends on `nx` only |
| `soundml-io` | audio-file decoding and encoding, resampling |

## Planned Features
 - A fast I/O for interacting with audio files
 - Feature extraction
 - Audio effects
   - Time stretching and pitch shifting
   - Filtering
     - IIR filters (Generic, Lowpass, Highpass)
     - Generic FIR filter implementation

## Building

SoundML is built with dune package management. Dune solves the dependencies, fetches
them, builds the OCaml compiler it needs and then builds the project. There is no
opam switch to create and nothing to install by hand.

You need dune 3.24 or later, which installs on its own:

```console
$ curl -fsSL https://get.dune.build/install | sh
```

plus the system libraries the satellite packages bind to. On Debian and Ubuntu:

```console
$ sudo apt-get install pkg-config libsndfile-dev
```

Then, from the root of the repository:

```console
$ dune pkg lock     # solve the dependencies and write dune.lock
$ dune build
$ dune runtest
```

`dune pkg lock` selects OCaml 5.5.0 and dune builds that compiler itself. The first
build therefore takes a while; later ones are served from `~/.cache/dune`. The lock
directory is not committed, so `dune pkg lock` has to be run once after cloning and
again whenever `dune-project` changes.

Developer tooling comes from dune as well, as *dev tools*: dune builds them from
source, at the version the repository asks for, outside the project's own dependency
solution. They are installed once:

```console
$ dune tools install ocamlformat   # the version recorded in .ocamlformat
$ dune tools install odoc
```

after which `dune fmt` and `dune build @doc` use them. Without the install step dune
falls back to whatever happens to be on `PATH`, which is usually built against a
different compiler and fails.

### The Raven pin

SoundML tracks a single, pinned revision of [Raven](https://github.com/raven-ml/raven)
rather than a released version of `nx`. The pin currently points at an integration
branch of the [gabyfle/raven](https://github.com/gabyfle/raven) fork that carries the
FFT and framing work SoundML relies on ahead of upstream review. The revision is
recorded in `dune-project`:

```
git+https://github.com/gabyfle/raven.git#69aa608f93ae6583a4d9f8d69c233c9f66bc0920
```

Raven is pre-release and its tensor API still moves between revisions, so moving the
pin is a deliberate change and not something a build picks up on its own. That
revision requires OCaml >= 5.5.0, which is therefore also SoundML's lower bound.

## License

Distributed under the Apache License Version 2.0. See LICENSE for more information.

## References

- **McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto** (2015). *librosa: Audio and music signal analysis in python.* In Proceedings of the 14th python in science conference, pp. 18-25.

- **Bellanger, M.** (2022). *Traitement numérique du signal. 10e édition.* Dunod.

- **Wang, L., Zhao, J., & Mortier, R.** (2022). *OCaml Scientific Computing*. Springer International Publishing eBooks. DOI: [10.1007/978-3-030-97645-3](https://doi.org/10.1007/978-3-030-97645-3)

- **Zoelzer, U.** (2002). *Dafx: Digital Audio Effects*. DOI: [10.1002/9781119991298](https://doi.org/10.1002/9781119991298)

- **Müller, M.** (2015). *Fundamentals of Music Processing*. Cambridge International Law Journal. DOI: [10.1007/978-3-319-21945-5](https://doi.org/10.1007/978-3-319-21945-5)

## Acknowledgements

* Logo generated with DALL-E by OpenAI
