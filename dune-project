(lang dune 3.17)

(name soundml)

(generate_opam_files true)

(source
 (github gabyfle/SoundML))

(authors "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(maintainers "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(license Apache-2.0)

(documentation https://soundml.dev)

(pin
 (url
  "git+https://github.com/raven-ml/raven.git#d9facbc227ddc80d644bdeff0d94cf76d6a0c07b")
 (package (name nx)))

(package
 (name soundml)
 (synopsis "An OCaml library to embed sound processing in your applications")
 (description
  "SoundML is a library built on top of Raven's Nx package to analyse sound files. It can read and write audio, extract features from audio files, and much more.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  (conf-sndfile :build)
  (conf-rubberband :build)
  ; Samplerate is already a requirement of rubberband
  (conf-samplerate :build)
  (dune-configurator :build)
  (nx
   (= dev))
  (odoc :with-doc)
  (alcotest :with-test)
  (yojson :with-test))
 (tags
  (sound spectrogram fourier digital-filters)))
