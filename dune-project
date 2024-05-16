(lang dune 3.15)

(name soundml)

(generate_opam_files true)

(source
 (github gabyfle/soundml))

(authors "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(maintainers "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(license Apache-2.0)

(documentation https://github.com/gabyfle/soundml)

(package
 (name soundml)
 (synopsis "A library to play with sound")
 (description
  "SoundML is a library built on top of Owl to analyse sounds files.")
 (depends
  (ocaml
   (>= 5.1.0))
  dune
  (owl
   (>= 1.1))
  (ffmpeg
   (>= 1.1.11)))
 (tags
  (sound gabyfle)))
