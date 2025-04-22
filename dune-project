(lang dune 3.15)

(name soundml)

(generate_opam_files true)

(source
 (github gabyfle/SoundML))

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
  (conf-sndfile :build)
  (conf-rubberband :build)
  (dune-configurator :build)
  (odoc :with-doc)
  (owl
   (>= 1.2))
  (ffmpeg
   (>= 1.2.1)))
 (tags
  (sound owl ffmpeg gabyfle)))
