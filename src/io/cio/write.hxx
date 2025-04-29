/*****************************************************************************/
/*                                                                           */
/*                                                                           */
/*  Copyright (C) 2023-2025                                                  */
/*    Gabriel Santamaria                                                     */
/*                                                                           */
/*                                                                           */
/*  Licensed under the Apache License, Version 2.0 (the "License");          */
/*  you may not use this file except in compliance with the License.         */
/*  You may obtain a copy of the License at                                  */
/*                                                                           */
/*    http://www.apache.org/licenses/LICENSE-2.0                             */
/*                                                                           */
/*  Unless required by applicable law or agreed to in writing, software      */
/*  distributed under the License is distributed on an "AS IS" BASIS,        */
/*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/*  See the License for the specific language governing permissions and      */
/*  limitations under the License.                                           */
/*                                                                           */
/*****************************************************************************/

#ifndef SOUNDML_WRITER_H
#define SOUNDML_WRITER_H

#include "common.hxx"

namespace SoundML
{
    namespace IO
    {

        class AudioWriter
        {
        private:
            SndfileHandle sndfile;
            sf_count_t nframes;

        public:
            AudioWriter(SndfileHandle file, sf_count_t nframes)
                : sndfile(file), nframes(nframes) {}

            /**
             * @brief Writes the given audio data to the file
             * @param data Pointer to the data to write
             * @return An std::expected<void, Error> containing the error code on failure
             */
            template <typename T>
            std::expected<void, Error> write(const T *data)
            {
                sf_count_t written = sndfile.writef(data, nframes);
                if (written != nframes)
                {
                    int err = sndfile.error() ? sndfile.error() : SF_ERR_SYSTEM;
                    return std::unexpected(Error(err, SNDFILE_ERR));
                }

                return std::expected<void, Error>{};
            }
        };
    } /* namespace SoundML::IO */
} /* namespace SoundML */

/**
 * @brief Writes the given audio data to the file
 * @param filename The name of the file to write to.
 * @param ba_data The data to write (Bigarray).
 * @param metadata The metadata of the audio data (number of frames, sample rate, channels, format).
 * @tparam T The type of the audio data (float or double).
 *
 * @return An std::expected<void, Error> containing the error code on failure
 */
template <typename T>
CAMLprim value caml_write_audio_file(value filename, value ba_data, value metadata)
{
    using namespace SoundML::IO;

    CAMLparam3(filename, ba_data, metadata);

    std::string filename_str = String_val(filename);
    sf_count_t nframes_val = Long_val(Field(metadata, 0));
    int sample_rate_val = Long_val(Field(metadata, 1));
    int channels_val = Long_val(Field(metadata, 2));
    int format_val = Long_val(Field(metadata, 3));

    SndfileHandle sndfile(filename_str, SFM_WRITE, format_val, channels_val, sample_rate_val);
    if (int err = sndfile.error(); err)
    {
        raise_caml_exception(Error(err, SNDFILE_ERR), filename_str);
    }

    AudioWriter writer(sndfile, nframes_val);
    T *data = (T *)Caml_ba_data_val(ba_data);

    auto result = writer.write(data);
    if (result.has_value())
        return Val_unit;
    else
    {
        Error err = result.error();
        raise_caml_exception(err, filename_str);
    }

    CAMLreturn(Val_unit);
}

#endif /* SOUNDML_WRITER_H */