(lang dune 3.18)

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
 (synopsis "An OCaml library to embed sound processing in your applications")
 (description
  "SoundML is a library built on top of Owl to analyse sounds files. It can read, write audio, extract various features from audio files and much more.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (conf-sndfile :build)
  (conf-rubberband :build)
  ; Samplerate is already a requirement of rubberband
  (conf-samplerate :build)
  (dune-configurator :build)
  (owl ; Version 1.2 is required to have the DC transform
   (>= 1.2))
  (odoc :with-doc)
  (alcotest :with-test)
  (yojson :with-test)
  (bisect_ppx
   (and
    :with-test
    (>= 2.5.0))))
 (tags
  (sound spectrogram fourier digital-filters)))
