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

#ifndef SOUNDML_READER_H
#define SOUNDML_READER_H

#include <vector>
#include <memory>

#include <cmath>
#include <cstring>
#include <soxr.h>

#include "common.hxx"

namespace SoundML
{
    namespace IO
    {
        /**
         * Abstract class for an audio reader
         */
        template <typename T>
        class AudioReader
        {
        public:
            virtual ~AudioReader() = default;
            virtual std::expected<sf_count_t, Error> process_whole(SndfileHandle &, T *) = 0;
        };

        template <typename T>
        class SndfileReader : public AudioReader<T>
        {
            sf_count_t nframes;
            int channels;
            int sample_rate;
            int format;

        public:
            SndfileReader(sf_count_t nframes, int channels, int sample_rate, int format)
                : nframes(nframes), channels(channels), sample_rate(sample_rate), format(format) {}

            std::expected<sf_count_t, Error> process_whole(SndfileHandle &sndfile, T *data)
            {
                size_t nsamples = static_cast<size_t>(nframes * channels);
                size_t size_in_bytes = nsamples * sizeof(T);

                intnat ndims = (channels > 1) ? 2 : 1;
                intnat dims[ndims];

                if (ndims == 1)
                    dims[0] = static_cast<intnat>(nframes);
                else
                {
                    dims[0] = static_cast<intnat>(nframes);
                    dims[1] = static_cast<intnat>(channels);
                }

                int type_flag = 0;
                if constexpr (std::is_same_v<T, float>)
                    type_flag = CAML_BA_FLOAT32;
                else if constexpr (std::is_same_v<T, double>)
                    type_flag = CAML_BA_FLOAT64;
                else
                    static_assert(!std::is_same_v<T, T>, "Unsupported type T for OCaml Bigarray conversion");

                T *start = data;
                T *read_buffer = new T[SOUNDML_BUFFER_SIZE * channels];
                if (read_buffer == nullptr)
                    return std::unexpected(Error(-1, SOUNDML_ERR));

                sf_count_t read_frames = 0;
                sf_count_t total_read = 0;

                caml_release_runtime_system();

                while ((read_frames = sndfile.readf(read_buffer, SOUNDML_BUFFER_SIZE)) > 0)
                {
                    size_t bytes_read = read_frames * channels * sizeof(T);
                    size_t sample_offset = total_read * channels;

                    const T *src_chunk_ptr = read_buffer;
                    T *dest_ptr = start + sample_offset;
                    std::memcpy(dest_ptr, src_chunk_ptr, bytes_read);

                    total_read += read_frames;
                }

                caml_acquire_runtime_system();

                if (int err = sndfile.error(); err)
                {
                    delete[] read_buffer;
                    return std::unexpected(Error(err, SNDFILE_ERR));
                }

                delete[] read_buffer;
                return total_read;
            }
        };

        /**
         * @brief Very light wrapper around SoX resample library that implements AudioReader
         * @tparam T The type of the data
         */
        template <typename T>
        class SoXrReader : public AudioReader<T>
        {
            soxr_error_t err;

            double target_sr;
            double input_sr;

            soxr_datatype_t in_t;
            soxr_datatype_t out_t;
            soxr_io_spec_t io_spec;
            soxr_runtime_spec_t runtime_spec;
            soxr_quality_spec_t quality_spec;

        public:
            SoXrReader(double out_sr,
                       double in_sr,
                       resampling_t quality,
                       unsigned threads = 1) : target_sr(out_sr),
                                               input_sr(in_sr)
            {
                in_t = (std::is_same_v<T, float>) ? SOXR_FLOAT32_I : SOXR_FLOAT64_I;
                /* soxr support outputing to planar format (split channels) */
                out_t = (std::is_same_v<T, float>) ? SOXR_FLOAT32_I : SOXR_FLOAT64_I;

                io_spec = soxr_io_spec(in_t, out_t);
                runtime_spec = soxr_runtime_spec(threads);

                /* SOXR_ROLLOFF_SMALL is the default value of soxr */
                quality_spec = soxr_quality_spec(get_recipe_type(quality), SOXR_ROLLOFF_SMALL);
            }

            /**
             * @brief Reads the whole file and resamples it according to the target sample rate with the correct resampler
             * @param sndfile The SndfileHandle to read from
             * @param data Pointer to the data to write
             */
            std::expected<sf_count_t, Error> process_whole(SndfileHandle &sndfile, T *data)
            {
                sf_count_t frames = std::ceil((sndfile.frames() * target_sr) / input_sr);
                int channels = sndfile.channels();

                size_t nsamples = static_cast<size_t>(frames * channels);
                size_t size_in_bytes = static_cast<size_t>(nsamples * sizeof(T));

                T *output = data;

                sf_count_t total_read = 0;      /* total number of frames read from the source file */
                sf_count_t total_generated = 0; /* total number of frames resampled by the resampler */

                soxr_t raw_resampler = (soxr_create(
                    input_sr,
                    target_sr,
                    (unsigned)sndfile.channels(),
                    &err,
                    &io_spec,
                    &quality_spec,
                    &runtime_spec));

                if (raw_resampler == nullptr)
                {
                    std::string error_msg = err ? std::string(err) : "Unknown soxr error";
                    return std::unexpected(Error(error_msg, SOXR_ERR));
                }

                auto soxr_deleter = [](soxr_t ptr)
                { if(ptr) soxr_delete(ptr); };
                std::unique_ptr<struct soxr, decltype(soxr_deleter)> resampler(raw_resampler, soxr_deleter);
                T *read_buffer = new T[SOUNDML_BUFFER_SIZE * channels];

                bool input_finished = false;
                while (true)
                {
                    size_t frames_read = 0;
                    soxr_in_t current_in = nullptr;
                    size_t current_ilen = 0;

                    if (!input_finished)
                    {
                        frames_read = sndfile.readf(read_buffer, SOUNDML_BUFFER_SIZE);

                        if (sndfile.error())
                        {
                            delete[] read_buffer;
                            return std::unexpected(Error(sndfile.error(), SNDFILE_ERR));
                        }

                        if (frames_read == 0)
                            input_finished = true; /* current_in == nullptr & current_ilen = 0 */
                        else
                        {
                            current_in = read_buffer;
                            current_ilen = frames_read;
                        }

                        total_read += frames_read;
                    }

                    size_t idone = 0;
                    size_t odone = 0;

                    size_t remaining_output_frames = frames - total_generated;

                    if (remaining_output_frames == 0 && !input_finished)
                    {
                        delete[] read_buffer;
                        return std::unexpected(Error("Output buffer insufficient based on estimate", SOUNDML_ERR));
                    }

                    err = soxr_process(
                        resampler.get(),
                        current_in,   /* will be NULL if input_finished (flushing) */
                        current_ilen, /* will be 0 if input_finished (flushing) */
                        &idone,
                        reinterpret_cast<soxr_out_t>(output), /* output buffer is the big array */
                        remaining_output_frames,
                        &odone);

                    if (err)
                    {
                        delete[] read_buffer;
                        return std::unexpected(Error(std::string(err), SOXR_ERR));
                    }

                    if (odone > 0)
                    {
                        output += static_cast<size_t>(odone) * channels;
                        total_generated += odone;
                    }

                    if (total_generated > frames)
                    {
                        delete[] read_buffer;
                        return std::unexpected(Error("Output buffer overflow detected after soxr_process", SOUNDML_ERR));
                    }

                    if (input_finished && odone == 0)
                        break;
                }

                /* this is the "real" number of frames we should have */
                size_t accurate_frames = std::ceil((total_read * target_sr) / input_sr);

                /* if we lost some samples due to the resampling, we're padding the BA with zeros on each dim */
                if (total_generated < accurate_frames)
                {
                    /* padding_frames is the number of samples we "lost" during resampling */
                    size_t padding_frames = accurate_frames - total_generated;
                    if (padding_frames > 0)
                    {
                        size_t padding_samples = padding_frames * channels;
                        std::fill_n(output, padding_samples, static_cast<T>(0));
                    }
                }

                delete[] read_buffer;
                return total_generated;
            }
        };
    } /* namespace SoundML::IO */
} /* namespace SoundML */

/**
 * @brief Reads an audio file and returns the data into a Bigarray.
 *
 * @param filename The name of the audio file to read.
 * @param res_typ The resampling type (from resampling_t).
 * @param sample_rate The sample rate we're targeting for resampling.
 * @tparam T The type of the audio data (float or double).
 *
 * @return A tuple containing the audio data and its metadata.
 */
template <typename T>
inline value caml_read_audio_file(value filename, value res_typ, value trgt_sr)
{
    CAMLparam0();
    CAMLlocal3(audio_array, audio_metadata, returns);

    using namespace SoundML::IO;
    std::string filename_str(String_val(filename));
    int trgt_sr_val = Long_val(trgt_sr);
    resampling_t resampling_type = static_cast<resampling_t>(Long_val(res_typ));

    SndfileHandle sndfile(filename_str);
    if (int err = sndfile.error(); err)
        raise_caml_exception(Error(err, SNDFILE_ERR), filename_str);

    if (sndfile.frames() <= 0 || sndfile.channels() <= 0 || sndfile.samplerate() <= 0 || sndfile.format() <= 0)
        raise_caml_exception(Error(SF_ERR_MALFORMED_FILE, SNDFILE_ERR), filename_str);

    sf_count_t nframes = sndfile.frames();
    sf_count_t padded_frames = nframes;
    int channels = sndfile.channels();
    int format = sndfile.format();
    int sample_rate = sndfile.samplerate();

    AudioReader<T> *reader = nullptr;
    bool resampling_required = resampling_type != RS_NONE && trgt_sr_val != sample_rate;

    if (resampling_required)
    { /* resampling has been required + file's sr != target sr */
        padded_frames = static_cast<sf_count_t>(std::ceil((nframes * (double)trgt_sr_val) / (double)sample_rate));
        reader = new SoXrReader<T>(static_cast<double>(trgt_sr_val), static_cast<double>(sample_rate), resampling_type);
    }
    else
    {
        trgt_sr = sample_rate;
        reader = new SndfileReader<T>(nframes, channels, sample_rate, format);
    }

    if (reader == nullptr)
    {
        sndfile.~SndfileHandle();
        raise_caml_exception(Error(-1, SOUNDML_ERR), filename_str);
    }

    AudioMetadata metadata{
        nframes, channels, trgt_sr_val, padded_frames, format};

    intnat ndims = metadata.channels > 1 ? 2 : 1;
    intnat dims[ndims];

    if (ndims == 1)
        dims[0] = static_cast<intnat>(metadata.padded_frames);
    else
    {
        dims[0] = static_cast<intnat>(metadata.padded_frames);
        dims[1] = static_cast<intnat>(metadata.channels);
    }

    int type_flag = 0;
    if constexpr (std::is_same_v<T, float>)
        type_flag = CAML_BA_FLOAT32;
    else if constexpr (std::is_same_v<T, double>)
        type_flag = CAML_BA_FLOAT64;
    else
        static_assert(!std::is_same_v<T, T>, "Unsupported type T for OCaml Bigarray conversion");

    /* memory will be managed by OCaml */
    audio_array = caml_ba_alloc(type_flag | CAML_BA_C_LAYOUT, ndims, NULL, dims);

    auto result = reader->process_whole(sndfile, static_cast<T *>(Caml_ba_data_val(audio_array)));
    if (!result.has_value())
    {
        Error err = result.error();
        raise_caml_exception(err, filename_str);
    }

    sf_count_t read_frames = result.value();
    metadata.frames = read_frames;

    audio_metadata = caml_alloc_tuple(4);

    Store_field(audio_metadata, 0, Val_long(metadata.frames));
    Store_field(audio_metadata, 1, Val_int(metadata.channels));
    Store_field(audio_metadata, 2, Val_int(metadata.sample_rate));
    Store_field(audio_metadata, 3, Val_int(metadata.format));

    returns = caml_alloc_tuple(2);
    Store_field(returns, 0, audio_array);
    Store_field(returns, 1, audio_metadata);
    return returns;
}

#endif /* SOUNDFILE_READER_H */
