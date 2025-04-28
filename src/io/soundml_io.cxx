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

#include <string>
#include <vector>
#include <expected>
#include <variant>
#include <memory>

#include <cmath>
#include <cstring>

#include <sndfile.hh>
#include <soxr.h>

extern "C"
{
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
}

#define BUFFER_SIZE 4096

namespace
{
    typedef enum
    {
        SNDFILE_ERR,
        SOXR_ERR,
        OTHER_ERR
    } ErrorType;

    using Error = std::pair<std::variant<int, std::string>, ErrorType>;

    /**
     * @brief Little helper to get a string out of an error code
     * @param err The error code
     * @return A string with the error message
     */
    std::string get_error_string(Error error)
    {
        std::variant<int, std::string> err_code = error.first;
        ErrorType typ = error.second;
        switch (typ)
        {
        case SNDFILE_ERR:
            return std::string("sndfile: ") + sf_error_number(std::get<int>(err_code));
        case SOXR_ERR:
            return std::string("soxr: ") + std::get<std::string>(err_code);
        case OTHER_ERR:
            return std::string("soundml: ") + std::get<std::string>(err_code);
        default:
            break;
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
        sf_count_t padded_frames;
        int format;
    };

    using AudioData = std::pair<value, AudioMetadata>;

    typedef enum
    {
        RS_NONE = 0,
        RS_SOXR_QQ,
        RS_SOXR_LQ,
        RS_SOXR_MQ,
        RS_SOXR_HQ,
        RS_SOXR_VHQ,
        /* TODO: implement these resamplers */
        RS_SCR_LINEAR,
        RS_SINC_BEST_QUALITY,
        RS_SINC_MEDIUM_QUALITY,
        RS_SINC_FASTEST,
        RS_ZERO_ORDER_HOLD,
        RS_SRC_LINEAR
    } resampling_t;

    /**
     * @brief Helper to map resampling_t to SOXR resampling recipes
     * @param type The resampling type
     */
    unsigned long get_recipe_type(resampling_t type)
    {

        switch (type)
        {
        case RS_SOXR_VHQ:
            return SOXR_VHQ;
        case RS_SOXR_HQ:
            return SOXR_HQ;
        case RS_SOXR_MQ:
            return SOXR_MQ;
        case RS_SOXR_LQ:
            return SOXR_LQ;
        default:
            return SOXR_VHQ;
        }
    }

    /**
     * Abstract class for an audio reader
     */
    class AudioReader
    {
    private:
        SndfileHandle sndfile;

    public:
        virtual ~AudioReader() = default;
        virtual std::expected<AudioData, Error> process_whole(value &) = 0;
    };

    template <typename T>
    class SndfileReader : public AudioReader
    {
        SndfileHandle sndfile;

        sf_count_t nframes;
        int channels;
        int sample_rate;
        int format;

    public:
        SndfileReader(SndfileHandle file)
            : sndfile(file)
        {
            nframes = sndfile.frames();
            channels = sndfile.channels();
            sample_rate = sndfile.samplerate();
            format = sndfile.format();
        }

        std::expected<AudioData, Error> process_whole(value &bigarray)
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

            /* memory is managed by OCaml */
            bigarray = caml_ba_alloc(type_flag | CAML_BA_C_LAYOUT, ndims, NULL, dims);
            T *read_buffer = (T *)std::aligned_alloc(BUFFER_SIZE, BUFFER_SIZE * sizeof(T) * channels);
            if (read_buffer == nullptr)
                return std::unexpected(Error(-1, OTHER_ERR));

            sf_count_t read_frames = 0;
            sf_count_t total_read = 0;

            T *start = static_cast<T *>(Caml_ba_data_val(bigarray));

            while ((read_frames = sndfile.readf(read_buffer, BUFFER_SIZE)) > 0)
            {
                size_t bytes_read = read_frames * channels * sizeof(T);
                size_t sample_offset = total_read * channels;

                const T *src_chunk_ptr = read_buffer;
                T *dest_ptr = start + sample_offset;
                std::memcpy(dest_ptr, src_chunk_ptr, bytes_read);

                total_read += read_frames;
            }

            if (int err = sndfile.error(); err)
            {
                std::free(read_buffer);
                return std::unexpected(Error(err, SNDFILE_ERR));
            }

            std::free(read_buffer);
            read_buffer = nullptr;

            AudioMetadata metadata{total_read, channels, sample_rate, total_read, format};
            return AudioData(bigarray, metadata);
        }
    };

    /**
     * @brief Very light wrapper around SoX resample library that implements AudioReader
     * @tparam T The type of the data
     */
    template <typename T>
    class SoXrReader : public AudioReader
    {
        SndfileHandle sndfile;
        soxr_error_t err;

        double target_sr;
        double input_sr;

        soxr_datatype_t in_t;
        soxr_datatype_t out_t;
        soxr_io_spec_t io_spec;
        soxr_runtime_spec_t runtime_spec;
        soxr_quality_spec_t quality_spec;

    public:
        bool fix{true};

        SoXrReader(SndfileHandle file,
                   double out_sr,
                   resampling_t quality,
                   unsigned threads = 1) : sndfile(file), target_sr(out_sr), input_sr((double)sndfile.samplerate())
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
         * @param bigarray The bigarray to fill with the read and resampled data.
         */
        std::expected<AudioData, Error> process_whole(value &bigarray) /* TODO: split this func in smaller funcsw */
        {
            /* NOTE: for some formats (like MP3), this is just an estimate, not an accurate number */
            sf_count_t frames = std::ceil((sndfile.frames() * target_sr) / input_sr);
            int channels = sndfile.channels();

            intnat ndims = (sndfile.channels() > 1) ? 2 : 1;
            intnat dims[ndims];

            /* TODO: Manage if *fix* is set to false */
            if (ndims == 1)
                dims[0] = static_cast<intnat>(frames);
            else
            {
                dims[0] = static_cast<intnat>(frames);
                dims[1] = static_cast<intnat>(channels);
            }

            int type_flag = 0;
            if constexpr (std::is_same_v<T, float>)
                type_flag = CAML_BA_FLOAT32;
            else if constexpr (std::is_same_v<T, double>)
                type_flag = CAML_BA_FLOAT64;
            else
                static_assert(!std::is_same_v<T, T>, "Unsupported type T for OCaml Bigarray conversion");

            /* memory is managed by OCaml */
            bigarray = caml_ba_alloc(type_flag | CAML_BA_C_LAYOUT, ndims, NULL, dims);
            T *ba_output = static_cast<T *>(Caml_ba_data_val(bigarray));

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
            T *read_buffer = (T *)std::aligned_alloc(BUFFER_SIZE, BUFFER_SIZE * sizeof(T) * channels);

            bool input_finished = false;
            while (true)
            {
                size_t frames_read = 0;
                soxr_in_t current_in = nullptr;
                size_t current_ilen = 0;

                if (!input_finished)
                {
                    frames_read = sndfile.readf(read_buffer, BUFFER_SIZE);

                    if (sndfile.error())
                    {
                        std::free(read_buffer);
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
                    std::free(read_buffer);
                    return std::unexpected(Error("Output buffer insufficient based on estimate", OTHER_ERR));
                }

                err = soxr_process(
                    resampler.get(),
                    current_in,   /* will be NULL if input_finished (flushing) */
                    current_ilen, /* will be 0 if input_finished (flushing) */
                    &idone,
                    reinterpret_cast<soxr_out_t>(ba_output), /* output buffer is the big array */
                    remaining_output_frames,
                    &odone);

                if (err)
                {
                    std::free(read_buffer);
                    return std::unexpected(Error(std::string(err), SOXR_ERR));
                }

                if (odone > 0)
                {
                    ba_output += static_cast<size_t>(odone) * channels;
                    total_generated += odone;
                }

                if (total_generated > frames)
                {
                    std::free(read_buffer);
                    return std::unexpected(Error("Output buffer overflow detected after soxr_process", OTHER_ERR));
                }

                if (input_finished && odone == 0)
                    break;
            }

            /* this is the "real" number of frames we should have */
            size_t accurate_frames = std::ceil((total_read * target_sr) / input_sr);

            /* if we lost some samples due to the resampling, we're padding the BA with zeros on each dim */
            if (fix && total_generated < accurate_frames)
            {
                /* padding_frames is the number of samples we "lost" during resampling */
                size_t padding_frames = accurate_frames - total_generated;
                if (padding_frames > 0)
                {
                    size_t padding_samples = padding_frames * channels;
                    std::fill_n(ba_output, padding_samples, static_cast<T>(0));
                }
            }

            std::free(read_buffer);
            read_buffer = nullptr;

            AudioMetadata metadata;
            metadata.frames = total_generated; /* we can then access on the real number of resampled data on OCaml side */
            metadata.channels = channels;
            metadata.sample_rate = static_cast<int>(target_sr);
            metadata.padded_frames = fix ? accurate_frames : total_generated;
            metadata.format = sndfile.format();

            return AudioData(bigarray, metadata);
        }
    };

    /**
     * Reads an audio file and returns the data into a vector.
     *
     * @param filename The name of the audio file to read.
     * @param audio_data OCaml value that'll hold the audio data (Bigarray).
     * @param res_typ The resampling type (from resampling_t).
     * @param sample_rate The sample rate we're targeting for resampling.
     * @param converter_type Converter type we need to use (from SRC).
     * @tparam T The type of the audio data (float or double).
     *
     * @return A std::expected<AudioData> containing the audio data on success, an std::unexpected<int> containing the error code on failure.
     */
    template <typename T>
    std::expected<AudioData, Error> read_audio_file(
        const std::string &filename, const resampling_t &res_typ, const int &sample_rate, value &audio_data)
    {
        SndfileHandle sndfile(filename);
        if (int err = sndfile.error(); err)
            return std::unexpected(Error(err, SNDFILE_ERR));

        if (sndfile.frames() <= 0 || sndfile.channels() <= 0 || sndfile.samplerate() <= 0 || sndfile.format() <= 0)
            return std::unexpected(Error(SF_ERR_MALFORMED_FILE, SNDFILE_ERR));

        AudioReader *reader = nullptr;
        if (res_typ != RS_NONE && sample_rate != sndfile.samplerate()) /* resampling has been required + file's sr != target sr */
            reader = new SoXrReader<T>(sndfile, static_cast<double>(sample_rate), res_typ);
        else
            reader = new SndfileReader<T>(sndfile);

        if (reader == nullptr)
            return std::unexpected(Error(-1, OTHER_ERR));

        auto result = reader->process_whole(audio_data);

        delete reader;

        return result;
    }

    /**
     * @brief Reads an audio file and returns the data into a Bigarray.
     *
     * @param filename The name of the audio file to read.
     * @param sample_rate The sample rate we're targeting for resampling.
     * @tparam T The type of the audio data (float or double).
     *
     * @return A tuple containing the audio data and its metadata.
     */
    template <typename T>
    CAMLprim value caml_read_audio_file(value filename, value res_typ, value sample_rate)
    {
        CAMLparam3(filename, res_typ, sample_rate);
        CAMLlocal4(caml_buffer, audio_array, audio_metadata, returns);

        std::string filename_str(String_val(filename));
        int sample_rate_val = Long_val(sample_rate);
        resampling_t resampling_type = static_cast<resampling_t>(Long_val(res_typ));

        auto result = read_audio_file<T>(filename_str, resampling_type, sample_rate_val, audio_array);
        if (!result.has_value())
        {
            Error err = result.error();
            std::string error_str = get_error_string(err);
            caml_failwith(error_str.c_str());
        }

        AudioData audio_data = std::move(result.value());

        const value &audio_samples = audio_data.first;
        const AudioMetadata &metadata = audio_data.second;

        audio_metadata = caml_alloc_tuple(5);

        Store_field(audio_metadata, 0, Val_long(metadata.frames));
        Store_field(audio_metadata, 1, Val_int(metadata.channels));
        Store_field(audio_metadata, 2, Val_int(metadata.sample_rate));
        Store_field(audio_metadata, 3, Val_int(metadata.padded_frames));
        Store_field(audio_metadata, 4, Val_int(metadata.format));

        returns = caml_alloc_tuple(2);
        Store_field(returns, 0, audio_samples);
        Store_field(returns, 1, audio_metadata);
        CAMLreturn(returns);
    }
} /* namespace */

extern "C"
{
    CAMLprim value caml_read_audio_file_f32(value filename, value res_typ, value sample_rate)
    {
        return caml_read_audio_file<float>(filename, res_typ, sample_rate);
    }

    CAMLprim value caml_read_audio_file_f64(value filename, value res_typ, value sample_rate)
    {
        return caml_read_audio_file<double>(filename, res_typ, sample_rate);
    }
}
