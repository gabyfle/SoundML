(lang dune 3.24)

(name soundml)

(generate_opam_files true)

(opam_file_location inside_opam_directory)

(implicit_transitive_deps false)

(source
 (github gabyfle/SoundML))

(authors "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(maintainers "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(license Apache-2.0)

(documentation https://soundml.dev)

(pin
 (url
  "git+https://github.com/gabyfle/raven.git#13ea63112e1859a9697c889da41f063763b3704b")
 (package
  (name nx))
 (package
  (name compress))
 (package
  (name rig)))

; Raven pins windtrap to a commit and thumper pins it to v0.2.0; dune refuses
; two assignments of one package, so the root settles it on Raven's commit.

(pin
 (url
  "git+https://github.com/invariant-hq/windtrap.git#7be03df651610949a87ed3bb797984015e9691b9")
 (package
  (name windtrap))
 (package
  (name ppx_windtrap)))

; thumper is not published to the opam repository yet: pin it the way nx is
; pinned, to a fixed commit so the lock is reproducible.

(pin
 (url
  "git+https://github.com/invariant-hq/thumper.git#5410a9c7fad23ae853d71896378aa815fc9a1fc8")
 (package
  (name thumper)))

(package
 (name soundml)
 (dir soundml)
 (synopsis "An OCaml library to embed sound processing in your applications")
 (description
  "SoundML is a library built on top of Raven's Nx package to analyse sound files. It can read and write audio, extract features from audio files, and much more.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  (nx
   (= dev))
  (rig
   (= dev))
  (odoc :with-doc)
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (thumper :with-test)
  (yojson :with-test))
 (tags
  (sound spectrogram fourier digital-filters)))

(package
 (name soundml-io)
 (dir soundml-io)
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
  (rig
   (= dev))
  (conf-pkg-config :build)
  (conf-sndfile :build)
  (dune-configurator :build)
  (odoc :with-doc)
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (thumper :with-test)
  (yojson :with-test))
 (tags
  (sound audio file-io)))
