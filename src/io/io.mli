(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2023-2025                                                  *)
(*    Gabriel Santamaria                                                     *)
(*                                                                           *)
(*                                                                           *)
(*  Licensed under the Apache License, Version 2.0 (the "License");          *)
(*  you may not use this file except in compliance with the License.         *)
(*  You may obtain a copy of the License at                                  *)
(*                                                                           *)
(*    http://www.apache.org/licenses/LICENSE-2.0                             *)
(*                                                                           *)
(*  Unless required by applicable law or agreed to in writing, software      *)
(*  distributed under the License is distributed on an "AS IS" BASIS,        *)
(*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *)
(*  See the License for the specific language governing permissions and      *)
(*  limitations under the License.                                           *)
(*                                                                           *)
(*****************************************************************************)

(** Audio file input/output operations.
    
    This module provides functions for reading and writing audio files with
    support for various formats, resampling, and channel conversion. It serves
    as the primary interface for loading audio data into SoundML and saving
    processed results back to disk. *)

(** {2 Exceptions} *)

(** Raised when a requested audio file cannot be found on the filesystem.
    The string contains the path that was not found. *)
exception File_not_found of string

(** Raised when attempting to read a file with an unsupported or corrupted format,
    or when trying to write in an unsupported format. The string contains
    details about the format issue. *)
exception Invalid_format of string

(** Raised when audio resampling fails. This can occur due to invalid sample
    rates, memory issues, or internal resampling library errors. *)
exception Resampling_error of string

(** {2 Types} *)

(** Internal resampling type for C++ backend *)
type resampling_t =
  | NONE  (** Skip resampling entirely - use original sample rate. *)
  | SOXR_QQ  (** Fast cubic interpolation - good for real-time applications. *)
  | SOXR_LQ  (** 16-bit quality with larger rolloff - faster processing. *)
  | SOXR_MQ  (** 16-bit quality with medium rolloff - balanced speed/quality. *)
  | SOXR_HQ  (** High quality resampling - recommended for most applications. *)
  | SOXR_VHQ  (** Maximum quality resampling - slower but best results. *)

(** {2 Audio File Reading} *)

val read :
     ?res_typ:resampling_t
  -> ?sample_rate:int
  -> ?mono:bool
  -> 'dev Rune.device
  -> (float, 'a) Rune.dtype
  -> string
  -> (float, 'a, 'dev) Rune.t * int
(** [read device dtype filename] reads an audio file and returns audio data with sample rate.

   Loads audio data from various file formats with optional resampling and
   channel conversion. The function automatically handles format detection
   and provides high-quality resampling when needed.

   @param resampling Resampling quality (default: High_quality)
   @param target_sample_rate Target sample rate in Hz (default: 22050)
   @param force_mono Convert to mono by averaging channels (default: true)
   @param device Device on which the resulting tensor should be loaded.
   @param dtype Data type for audio samples (Rune.float32 or Rune.float64)
   @param filename Path to the audio file to read
   @return Tuple of (audio_tensor, actual_sample_rate)

   @raise File_not_found if the file doesn't exist
   @raise Invalid_audio_format if the file format is unsupported or corrupted
   @raise Resampling_error if resampling fails

   {3 Examples}

   Basic audio loading:
   {[
     let audio_data, sample_rate = IO.read ~device:Rune.ocaml 
       Rune.float32 
       ~filename:"audio.wav" in
     (* audio_data is mono float32 tensor at 22050 Hz *)
   ]}

   Load stereo audio without resampling:
   {[
     let audio_data, sample_rate = IO.read_audio_file 
       ~resampling:No_resampling
       ~force_mono:false
       Rune.float64 
       ~filename:"stereo.flac" in
     (* Preserves original sample rate and stereo channels *)
   ]}

   Load with specific device and sample rate:
   {[
     let audio_data, sample_rate = IO.read
       ~target_sample_rate:44100
       ~resampling:Very_high_quality
       Rune.float32 
       ~filename:"music.mp3" in
     (* High-quality resampling to 44.1 kHz on Metal device *)
   ]}

   {3 Supported Formats}

   SoundML uses libsndfile for audio I/O, supporting:
   - WAV (uncompressed PCM)
   - FLAC (lossless compression)
   - OGG Vorbis (lossy compression)
   - AIFF (uncompressed PCM)
   - MP3 (via external libraries)
   - Many other formats - see {{:https://libsndfile.github.io/libsndfile/formats.html}libsndfile documentation}

   {3 Performance Notes}

   - Use float32 for better performance unless you need float64 precision
   - No_resampling is fastest when you can work with the original sample rate *)

(** {2 Audio File Writing} *)

val write :
  ?format:Aformat.t -> string -> (float, 'a, 'dev) Rune.t -> int -> unit
(** [write filename audio_data sample_rate] writes audio data to a file.

   Saves audio tensor data to various file formats with automatic format
   detection based on file extension or explicit format specification.

   @param format Output format (default: Auto_detect from filename extension)
   @param filename Output file path
   @param audio_data Audio tensor to write (1D for mono, 2D for multi-channel)
   @param sample_rate Sample rate of the audio data in Hz

   @raise Invalid_argument if sample_rate <= 0
   @raise Invalid_argument if audio_data has invalid shape (> 2D)
   @raise Invalid_audio_format if the output format is unsupported

   {3 Examples}

   Basic audio writing (format auto-detected):
   {[
     IO.write 
       "output.wav"
       processed_audio
       22050
     (* Writes as WAV format based on .wav extension *)
   ]}

   Write with explicit format:
   {[
     IO.write 
       ~format:FLAC
       "output.audio"
       audio_tensor
       44100
     (* Writes as FLAC despite .audio extension *)
   ]}

  Write stereo audio:
   {[
     let stereo_audio = (* 2D tensor: [2; n_samples] *) in
     IO.write 
       "stereo.wav"
       stereo_audio
       48000
     (* Writes 2-channel audio *)
   ]}

   {3 Format Support}

   - WAV: Uncompressed, high quality, widely supported
   - FLAC: Lossless compression, smaller files than WAV
   - OGG: Lossy compression, good quality/size ratio
   - MP3: Lossy compression, universal compatibility (if available)
   - AIFF: Uncompressed, common on macOS

   {3 Performance Notes}

   - WAV writing is fastest (no compression)
   - FLAC provides good compression with minimal CPU overhead
   - Use appropriate bit depth for your application (float32 vs float64) *)
