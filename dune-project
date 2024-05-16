(lang dune 3.15)

(name soundml)

(generate_opam_files true)

(source
 (github gabyfle/soundml))

(authors "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(maintainers "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(license LICENSE)

(documentation https://url/to/documentation)

(package
 (name soundml)
 (synopsis "A library to play with sound")
 (description
  "SoundML is a library built on top of Owl to analyse sounds files.")
 (depends
  (ocaml
   (>= 5.2.0))
  (dune
   (>= 3.15))
  (owl
   (>= 1.1))
  (ffmpeg
   (>= 1.1.11)))
 (tags
  (sound gabyfle)))
