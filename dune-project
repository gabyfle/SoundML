(lang dune 3.15)

(name soundml)

(generate_opam_files true)

(source
 (github gabyfle/soundml))

(authors "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(maintainers "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(license Apache-2.0)

(documentation https://soundml.gabyfle.dev)

(package
 (name soundml)
 (synopsis
  "An OCaml library to embed sound processing in your applications.")
 (description
  "SoundML is a library built on top of Owl to analyse sounds files. It can read, write audio, extract various features from audio files and much more.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (conf-pkg-config :build)
  (dune-configurator :build)
  (owl
   (>= 1.1))
  (ffmpeg
   (>= 1.2.1)))
 (tags
  (sound owl ffmpeg gabyfle)))
