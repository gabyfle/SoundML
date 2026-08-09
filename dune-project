(lang dune 3.24)

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
  "git+https://github.com/raven-ml/raven.git#5f9ff1a397b915b5854670940bba1f3fa11c01f6")
 (package
  (name nx)))

; thumper is not published to the opam repository yet: pin it the way nx is
; pinned, to a fixed commit so the lock is reproducible.

(pin
 (url
  "git+https://github.com/invariant-hq/thumper.git#0d42dc6964e30525b0595b6d5be04a1eda76c90e")
 (package
  (name thumper)))

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
  (windtrap :with-test)
  (thumper :with-test)
  (yojson :with-test))
 (tags
  (sound spectrogram fourier digital-filters)))

(package
 (name soundml-io)
 (synopsis "Audio-file I/O for SoundML")
 (description
  "SoundML I/O decodes and encodes audio with libsndfile. Load-time resampling is Soundml.Resample, fed at decode time.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  (soundml
   (= :version))
  (nx
   (= dev))
  (conf-pkg-config :build)
  (conf-sndfile :build)
  (dune-configurator :build)
  (odoc :with-doc)
  (windtrap :with-test)
  (thumper :with-test)
  (yojson :with-test))
 (tags
  (sound audio file-io)))
