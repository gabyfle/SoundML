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

#include <rubberband/RubberBandStretcher.h>
#include <expected>
#include <cmath>

extern "C"
{
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
}

namespace SoundML
{
    namespace Effects
    {
        namespace Time
        {
            using namespace RubberBand;

            /**
             * @brief Stretches the input audio data using RubberBand.
             *
             * @param intput The input audio data pointer.
             * @param output The output audio data OCaml value.
             * @param samples The number of samples in the input data.
             * @param sample_rate The sample rate of the input data.
             * @param channels The number of channels in the input data.
             * @param config The RubberBand configuration options.
             * @param time_ratio The time ratio for stretching.
             * @param pitch_scale The pitch scale for stretching.
             *
             * @return The stretched audio data.
             */
            std::expected<value, std::string> offline_stretch(
                float *intput,
                value output,
                size_t samples,
                size_t sample_rate,
                size_t channels,
                RubberBandStretcher::Options config,
                double time_ratio,
                double pitch_scale)
            {
                RubberBandStretcher stretcher(sample_rate, channels, config, time_ratio, pitch_scale);

                stretcher.setExpectedInputDuration(samples);

                /* we have access to the whole input, so we can feed it directly into the stretcher */
                stretcher.study(&intput, samples, true);
                stretcher.process(&intput, samples, true); /* Rubberband expect deinterleaved samples */

                size_t per_channel_size = stretcher.available();

                intnat ndims = (channels > 1) ? 2 : 1;
                intnat dims[ndims];

                if (ndims == 1)
                    dims[0] = static_cast<intnat>(per_channel_size);
                else
                {
                    dims[0] = static_cast<intnat>(channels); /* we're going to get the data directly deinterleaved */
                    dims[1] = static_cast<intnat>(per_channel_size);
                }

                /* memory is managed by OCaml */
                output = caml_ba_alloc(CAML_BA_FLOAT32 | CAML_BA_C_LAYOUT, ndims, NULL, dims);

                size_t retrieved = stretcher.retrieve((float *const *)&Caml_ba_data_val(output), per_channel_size);
                if (retrieved != per_channel_size)
                {
                    std::string error_msg = "Rubberband error: retrieved " + std::to_string(retrieved) + " samples, expected " + std::to_string(per_channel_size);
                    return std::unexpected(error_msg);
                }

                return output;
            }

        } /* namespace Time */
    } /* namespace Effects */
} /* namespace SoundML */

extern "C"
{
    CAMLprim value caml_rubberband_stretch(value input, value params)
    {
        using namespace SoundML::Effects::Time;
        CAMLparam2(input, params);
        CAMLlocal1(output);

        size_t samples_val = Long_val(Field(params, 0));
        size_t sample_rate_val = Long_val(Field(params, 1));
        size_t channels_val = Long_val(Field(params, 2));
        RubberBandStretcher::Options config_val = static_cast<RubberBandStretcher::Options>(Long_val(Field(params, 3)));
        double time_ratio_val = Double_val(Field(params, 4));
        double pitch_scale_val = Double_val(Field(params, 5));

        float *input_data = (float *)Caml_ba_data_val(input);

        auto result = offline_stretch(input_data, output, samples_val, sample_rate_val, channels_val, config_val, time_ratio_val, pitch_scale_val);

        if (!result.has_value())
        {
            caml_failwith(result.error().c_str());
        }

        CAMLreturn(result.value());
    }
}
