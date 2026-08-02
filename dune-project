(lang dune 3.17)

(name soundml)

(generate_opam_files true)

(implicit_transitive_deps false)

(source
 (github gabyfle/SoundML))

(authors "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(maintainers "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(license Apache-2.0)

(documentation https://soundml.dev)

(pin
 (url
  "git+https://github.com/raven-ml/raven.git#d9facbc227ddc80d644bdeff0d94cf76d6a0c07b")
 (package
  (name nx)))

(package
 (name soundml)
 (synopsis "An OCaml library to embed sound processing in your applications")
 (description
  "SoundML is a library built on top of Raven's Nx package to analyse sound files. It can read and write audio, extract features from audio files, and much more.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  (nx
   (= dev))
  (odoc :with-doc)
  (alcotest :with-test)
  (yojson :with-test))
 (tags
  (sound spectrogram fourier digital-filters)))

(package
 (name soundml-io)
 (allow_empty)
 (synopsis "Audio-file I/O and resampling for SoundML")
 (description
  "SoundML I/O decodes and encodes audio with libsndfile and optionally resamples it with SoXR.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  (soundml
   (= :version))
  (nx
   (= dev))
  (conf-c++ :build)
  (conf-pkg-config :build)
  (conf-sndfile :build)
  (conf-samplerate :build)
  (dune-configurator :build)
  (odoc :with-doc)
  (alcotest :with-test))
 (tags
  (sound audio file-io resampling)))

(package
 (name soundml-rubberband)
 (allow_empty)
 (synopsis "Rubber Band time and pitch effects for SoundML")
 (description
  "SoundML Rubber Band provides offline time stretching and pitch shifting over SoundML audio values.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  (soundml
   (= :version))
  (nx
   (= dev))
  (conf-c++ :build)
  (conf-pkg-config :build)
  (conf-rubberband :build)
  (dune-configurator :build)
  (odoc :with-doc)
  (alcotest :with-test))
 (tags
  (sound audio time-stretching pitch-shifting)))
