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

#include <vector>
#include <string>
#include <tuple>
#include <expected>
#include <limits>
#include <cmath>
#include <cstring>

#include <sndfile.hh>
#include <samplerate.h>

extern "C"
{
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/custom.h>
}

typedef enum
{
    SNDFILE_ERR,
    SRC_ERR,
    OTHER_ERR
} ErrorType;

using Error = std::tuple<int, ErrorType>;

/**
 * @brief Little helper to get a string out of an error code
 * @param err The error code
 * @return A string with the error message
 */
inline std::string get_error_string(Error error)
{
    int err_code = std::get<0>(error);
    ErrorType typ = std::get<1>(error);
    switch (typ)
    {
    case SNDFILE_ERR:
        return std::string("sndfile: ") + sf_error_number(err_code);
    case SRC_ERR:
        return std::string("samplerate: ") + src_strerror(err_code);
    case OTHER_ERR:
        return std::string("Unknown error");
    }

    return std::string("Unknown error");
}

/**
 * @brief Little structure holding audio metadata.
 */
struct AudioMetadata
{
    sf_count_t frames;
    int channels;
    int sample_rate;
    int format;
};

template <typename T>
using AudioData = std::tuple<std::vector<T>, AudioMetadata>;

/**
 * Reads an audio file and returns the data into a vector.
 *
 * @param filename The name of the audio file to read.
 * @param buffer_size The size of the buffer to read into.
 * @param trgt_sample_rate The sample rate we're targeting for resampling.
 * @param converter_type Converter type we need to use (from SRC).
 * @tparam T The type of the audio data (float or double).
 *
 * @return A std::expected<AudioData> containing the audio data on success, an std::unexpected<int> containing the error code on failure.
 */
template <typename T>
std::expected<AudioData<T>, Error> read_audio_file(
    const std::string &filename,
    sf_count_t buffer_size = 1024,
    int trgt_sample_rate = 22050,
    int converter_type = SRC_LINEAR)
{
    SndfileHandle sndfile(filename);
    if (int err = sndfile.error(); err)
        return std::unexpected(Error(err, SNDFILE_ERR));

    const sf_count_t nat_frames = sndfile.frames();
    const int channels = sndfile.channels();
    const int nat_sample_rate = sndfile.samplerate();
    const int format = sndfile.format();

    if (nat_frames <= 0 || channels <= 0 || nat_sample_rate <= 0 || format <= 0)
        return std::unexpected(Error(SF_ERR_MALFORMED_FILE, SNDFILE_ERR));

    bool needs_resampling = (trgt_sample_rate != nat_sample_rate);
    SRC_STATE *src_state = nullptr;
    SRC_DATA src_data;
    int src_error = 0;
    double src_ratio = 1.0;

    std::vector<float> resample_buffer;
    std::vector<float> processed_audio;
    std::vector<float> read_buffer(buffer_size * channels);

    if (needs_resampling)
    {
        src_ratio = static_cast<double>(trgt_sample_rate) / nat_sample_rate;
        src_state = src_new(converter_type, channels, &src_error);
        if (src_state == nullptr)
            return std::unexpected(Error(src_error, SRC_ERR));
        size_t max_output_frames_per_chunk = static_cast<size_t>(std::ceil(buffer_size * src_ratio)) + 10;
        size_t min_buffer_size = channels * 16;
        resample_buffer.resize(std::max(max_output_frames_per_chunk * channels, min_buffer_size));

        src_data.src_ratio = src_ratio;
        src_data.data_out = resample_buffer.data();
        src_data.output_frames = resample_buffer.size() / channels;
    }

    size_t estimated_samples = static_cast<size_t>(nat_frames * channels * src_ratio * 1.05);
    if (estimated_samples > 0 && estimated_samples < processed_audio.max_size())
        processed_audio.reserve(estimated_samples);
    else
        processed_audio.reserve(static_cast<size_t>(nat_frames * channels));

    sf_count_t read_frames = 0;
    sf_count_t total_read = 0;      /* number of frames read from the file */
    sf_count_t total_generated = 0; /* number of frames we "generated" during the resampling */

    while ((read_frames = sndfile.readf(read_buffer.data(), buffer_size)) > 0)
    {
        total_read += read_frames;

        if (needs_resampling)
        {
            src_data.data_in = read_buffer.data();
            src_data.input_frames = read_frames;
            src_data.end_of_input = 0;

            src_error = src_process(src_state, &src_data);
            if (src_error != 0)
            {
                src_delete(src_state);
                return std::unexpected(Error(src_error, SRC_ERR));
            }
            processed_audio.insert(processed_audio.end(),
                                   src_data.data_out,
                                   src_data.data_out + src_data.output_frames_gen * channels);
            total_generated += src_data.output_frames_gen;
        }
        else
        {
            processed_audio.insert(processed_audio.end(),
                                   read_buffer.data(),
                                   read_buffer.data() + read_frames * channels);
            total_generated += read_frames;
        }
    }

    if (int err = sndfile.error(); err)
    {
        if (src_state)
            src_delete(src_state);
        return std::unexpected(Error(err, SNDFILE_ERR));
    }

    /* flushing */
    if (needs_resampling)
    {
        src_data.data_in = nullptr;
        src_data.input_frames = 0;
        src_data.end_of_input = 1;

        do
        {
            src_error = src_process(src_state, &src_data);
            if (src_error != 0)
            {
                src_delete(src_state);
                return std::unexpected(Error(src_error, SRC_ERR));
            }

            if (src_data.output_frames_gen > 0)
            {
                processed_audio.insert(processed_audio.end(),
                                       src_data.data_out,
                                       src_data.data_out + src_data.output_frames_gen * channels);
                total_generated += src_data.output_frames_gen;
            }
            src_data.data_out = resample_buffer.data();
            src_data.output_frames = resample_buffer.size() / channels;

        } while (src_data.output_frames_gen > 0);

        src_delete(src_state);
        src_state = nullptr;
    }

    std::vector<T> audio;
    size_t n_samples = processed_audio.size();

    if constexpr (std::is_same_v<T, float>)
        audio = processed_audio;
    else if constexpr (std::is_same_v<T, double>)
    {
        audio.resize(n_samples);
        for (size_t i = 0; i < n_samples; ++i)
            audio[i] = static_cast<T>(processed_audio[i]);
        processed_audio.clear();
        processed_audio.shrink_to_fit();
    }
    else
        static_assert(!std::is_same_v<T, T>, "Unsupported type T for audio data");

    AudioMetadata metadata(total_generated, channels, trgt_sample_rate, format);
    return AudioData<T>(audio, metadata);
}

template <typename T>
CAMLprim value caml_read_audio_file(value filename, value buffer_size, value sample_rate)
{
    CAMLparam3(filename, buffer_size, sample_rate);
    CAMLlocal4(caml_buffer, audio_array, audio_metadata, returns);

    std::string filename_str(String_val(filename));
    sf_count_t buffer_size_val = Long_val(buffer_size);
    int sample_rate_val = Long_val(sample_rate);

    auto result = read_audio_file<T>(filename_str, buffer_size_val, sample_rate_val);
    if (!result.has_value())
    {
        Error err = result.error();
        std::string error_str = get_error_string(err);
        caml_failwith(error_str.c_str());
    }

    auto [audio_data, metadata] = result.value();
    size_t audio_size = audio_data.size();
    int ndims = (metadata.channels > 1) ? 2 : 1;
    intnat dims[ndims];
    if (ndims == 1)
    {
        dims[0] = static_cast<intnat>(metadata.frames);
    }
    else
    {
        dims[0] = static_cast<intnat>(metadata.channels);
        dims[1] = static_cast<intnat>(metadata.frames);
    }

    int type_flag = 0;
    if constexpr (std::is_same_v<T, float>)
        type_flag = CAML_BA_FLOAT32;
    else if constexpr (std::is_same_v<T, double>)
        type_flag = CAML_BA_FLOAT64;
    else
        static_assert(!std::is_same_v<T, T>, "Unsupported type T for OCaml Bigarray conversion");

    /* memory is managed by OCaml */
    audio_array = caml_ba_alloc(type_flag | CAML_BA_C_LAYOUT, ndims, NULL, dims);

    if (ndims == 1)
        std::memcpy(Caml_ba_data_val(audio_array), audio_data.data(), audio_size * sizeof(T));
    else /* we aim to have shape (channels, frames), so we need to deinterleave the samples */
    {
        /**
         * we could have used Owl's transpose function, but it'd mean allocating
         * a whole new bigarray and copying the data into it. By doing this, we
         * avoid the allocation.
         *
         * NOTE: transpose_ does exists, but it assumes that we have a ~out array
         * with the correct shape to transpose into, inplace.
         */
        const size_t nframes = metadata.frames;
        const int nchannels = metadata.channels;

        T *dest_data = static_cast<T *>(Caml_ba_data_val(audio_array));
        const T *src_data = audio_data.data();

        for (size_t f = 0; f < nframes; ++f)
        {
            for (int c = 0; c < nchannels; ++c)
            {
                size_t src_idx = f * nchannels + c; /* interleaved format */
                size_t dest_idx = c * nframes + f;  /* planar format */
                dest_data[dest_idx] = src_data[src_idx];
            }
        }
    }

    audio_metadata = caml_alloc_tuple(4);

    Store_field(audio_metadata, 0, Val_long(metadata.frames));
    Store_field(audio_metadata, 1, Val_int(metadata.channels));
    Store_field(audio_metadata, 2, Val_int(metadata.sample_rate));
    Store_field(audio_metadata, 3, Val_int(metadata.format));

    returns = caml_alloc_tuple(2);
    Store_field(returns, 0, audio_array);
    Store_field(returns, 1, audio_metadata);
    CAMLreturn(returns);
}

extern "C"
{
    CAMLprim value caml_read_audio_file_f32(value filename, value buffer_size, value sample_rate)
    {
        return caml_read_audio_file<float>(filename, buffer_size, sample_rate);
    }

    CAMLprim value caml_read_audio_file_f64(value filename, value buffer_size, value sample_rate)
    {
        return caml_read_audio_file<double>(filename, buffer_size, sample_rate);
    }
}