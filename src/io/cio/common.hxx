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

#define SOUNDML_BUFFER_SIZE 4096
#include <expected>
#include <variant>
#include <sndfile.hh>

extern "C" /* OCaml imports */
{
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/callback.h>
#include <caml/threads.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
}

#ifndef SOUNDML_IO_COMMON_H
#define SOUNDML_IO_COMMON_H
namespace SoundML
{
    namespace IO
    {
        typedef enum
        {
            SNDFILE_ERR,
            SOXR_ERR,
            SOUNDML_ERR
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
                return std::string(sf_error_number(std::get<int>(err_code)));
            case SOXR_ERR:
                return std::get<std::string>(err_code);
            case SOUNDML_ERR:
                return std::get<std::string>(err_code);
            default:
                break;
            }

            return std::string("Unknown error");
        }

        /**
         * @brief Raise the correct OCaml exception from the given Error
         * @param error The error to raise
         * @param filename The name of the file that caused the error
         */
        void raise_caml_exception(Error error, std::string filename)
        {
            ErrorType type = error.second;
            std::string error_string = SoundML::IO::get_error_string(error) + " in file " + filename;

#define GET_EXN_TAG(name) \
    (*caml_named_value(name))

            if (type == SNDFILE_ERR)
            {
                int err_code = std::get<int>(error.first);
                bool is_format_err = err_code == SF_ERR_UNRECOGNISED_FORMAT || err_code == SF_ERR_MALFORMED_FILE || err_code == SF_ERR_UNSUPPORTED_ENCODING;
                if (is_format_err)
                    caml_raise_with_string(GET_EXN_TAG("soundml.exn.invalid_format"), error_string.c_str());
                else /* it's SF_ERR_SYSTEM */
                    caml_raise_with_string(GET_EXN_TAG("soundml.exn.file_not_found"), error_string.c_str());
            }
            else if (type == SOXR_ERR)
                caml_raise_with_string(GET_EXN_TAG("soundml.exn.resampling_error"), error_string.c_str());
            else if (type == SOUNDML_ERR)
                caml_raise_with_string(GET_EXN_TAG("soundml.exn.internal_error"), error_string.c_str());
            else
                caml_raise_with_string(GET_EXN_TAG("soundml.exn.internal_error"), "Unknown internal error.");
#undef GET_EXN_TAG
        }

        /**
         * @brief Structure holding metadate related to an audio file
         * @param frames number of frames we read from the file
         * @param channels number of channels in the file
         * @param sample_rate sample-rate of the file (if a resampling has been asked, sample-rate equals the the asked sr)
         * @param padded_frames number of frames we padded with zeros
         * @param format format of the file (SF_FORMAT_* from libsndfile)
         */
        struct AudioMetadata
        {
            sf_count_t frames;
            int channels;
            int sample_rate;
            sf_count_t padded_frames;
            int format;
        };

        /**
         * @brief Enum that represents the resampling types
         * @note The SoX resampling types are defined in soxr.h
         */
        typedef enum
        {
            RS_NONE = 0, /* No resampling */
            RS_SOXR_QQ,  /* 'Quick' cubic interpolation. */
            RS_SOXR_LQ,  /* 'Low' 16-bit with larger rolloff. */
            RS_SOXR_MQ,  /* 'Medium' 16-bit with medium rolloff. */
            RS_SOXR_HQ,  /* 'High quality'. */
            RS_SOXR_VHQ, /* 'Very high quality'. */
            /* TODO: implement these resamplers */
            RS_SCR_LINEAR,
            RS_SINC_BEST_QUALITY,
            RS_SINC_MEDIUM_QUALITY,
            RS_SINC_FASTEST,
            RS_ZERO_ORDER_HOLD,
            RS_SRC_LINEAR
        } resampling_t;

        /**
         * @brief Get the (correct) SoX resampling type from the resampling_t enum
         * @param type The resampling type to convert
         * @return The SoX resampling type (SOXR_* from soxr.h)
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
    } /* namespace SoundML::IO */
} /* namespace SoundML */

#endif /* SOUNDML_IO_COMMON_H */
