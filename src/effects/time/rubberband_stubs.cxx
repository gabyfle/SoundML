/*****************************************************************************/
/*                                                                           */
/*                                                                           */
/*  Copyright (C) 2023                                                       */
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
#include <iostream> /* for debug purpose for the moment */
#include <cmath>

extern "C"
{
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>

    /**
     * Time-stretch a 32-bits audio signal using the RubberBand library.
     *
     * @param v_input The input audio signal given as an OCaml bigarray.
     * @param v_rate float that represents the time-stretching factor.
     * @param v_sample_rate Integer for the sample rate of the audio signal.
     * @param v_channels Integer for the number of channels of the audio signal.
     * @param v_config Integer for the configuration of the RubberBand stretcher.
     *
     * @return The time-stretched audio signal as a OCaml bigarray.
     */
    CAMLprim value caml_rubberband_time_stretch(value v_input, value v_rate, value v_sample_rate, value v_channels, value v_config)
    {
        CAMLparam5(v_input, v_rate, v_config, v_sample_rate, v_channels);
        CAMLlocal1(v_output);

        float *input_data = static_cast<float *>(Caml_ba_data_val(v_input));
        float rate = Double_val(v_rate);
        int sample_rate = Long_val(v_sample_rate);
        int channels = Long_val(v_channels);
        int config = Long_val(v_config);
        size_t num_samples = Caml_ba_array_val(v_input)->dim[0];

        RubberBand::RubberBandStretcher stretcher(sample_rate, channels, config);

        stretcher.setTimeRatio(rate);

        std::vector<const float *> inputChannels(channels);
        for (int c = 0; c < channels; ++c)
            inputChannels[c] = input_data + (c * num_samples);

        stretcher.study(inputChannels.data(), num_samples, true);
        stretcher.process(inputChannels.data(), num_samples, true);

        size_t outputSize = stretcher.available();
        if (outputSize == 0)
        {
            outputSize = stretcher.getSamplesRequired();
        }

        std::vector<std::vector<float>> outputBuffers(channels);
        std::vector<float *> outputChannels(channels);
        for (int c = 0; c < channels; ++c)
        {
            outputBuffers[c].resize(outputSize);
            outputChannels[c] = outputBuffers[c].data();
        }

        size_t retrievedSamples = stretcher.retrieve(outputChannels.data(), outputSize);

        size_t totalSamples = retrievedSamples * channels;
        float *output_data = static_cast<float *>(malloc(totalSamples * sizeof(float)));
        if (!output_data)
        {
            caml_failwith("Failed to allocate memory for output data");
        }

        if (channels > 1)
        {
            for (size_t i = 0; i < retrievedSamples; ++i)
                for (int c = 0; c < channels; ++c)
                    output_data[i * channels + c] = outputChannels[c][i];
        }
        else
        {
            std::copy(outputBuffers[0].begin(),
                      outputBuffers[0].begin() + retrievedSamples,
                      output_data);
        }

        intnat dims[channels];
        for (int c = 0; c < channels; ++c)
            dims[c] = static_cast<intnat>(totalSamples);

        v_output = caml_ba_alloc(CAML_BA_FLOAT32 | CAML_BA_C_LAYOUT, channels, output_data, dims);

        CAMLreturn(v_output);
    }

    /**
     * Pitch-shift a 32-bits audio signal using the RubberBand library.
     *
     * @param v_input The input audio signal given as an OCaml bigarray.
     * @param v_semitone int that represents the number of semitones to shift the pitch
     * @param v_sample_rate Integer for the sample rate of the audio signal.
     * @param v_channels Integer for the number of channels of the audio signal.
     * @param v_config Integer for the configuration of the RubberBand stretcher.
     *
     * @return The pitch-shifted audio signal as a OCaml bigarray.
     */

    CAMLprim value caml_rubberband_pitch_shift(value v_input, value v_semitone, value v_sample_rate, value v_channels, value v_config)
    {
        CAMLparam5(v_input, v_semitone, v_config, v_sample_rate, v_channels);
        CAMLlocal1(v_output);

        float *input_data = static_cast<float *>(Caml_ba_data_val(v_input));
        int semitone = Long_val(v_semitone);
        int sample_rate = Long_val(v_sample_rate);
        int channels = Long_val(v_channels);
        int config = Long_val(v_config);
        size_t num_samples = Caml_ba_array_val(v_input)->dim[0];

        RubberBand::RubberBandStretcher stretcher(sample_rate, channels, config);

        stretcher.setPitchScale(std::pow(2.0, semitone / 12.0));

        std::vector<const float *> inputChannels(channels);
        for (int c = 0; c < channels; ++c)
            inputChannels[c] = input_data + (c * num_samples);

        stretcher.study(inputChannels.data(), num_samples, true);
        stretcher.process(inputChannels.data(), num_samples, true);

        size_t outputSize = stretcher.available();

        if (outputSize == 0)
        {
            outputSize = stretcher.getSamplesRequired();
        }

        std::vector<std::vector<float>> outputBuffers(channels);
        std::vector<float *> outputChannels(channels);
        for (int c = 0; c < channels; ++c)
        {
            outputBuffers[c].resize(outputSize);
            outputChannels[c] = outputBuffers[c].data();
        }

        size_t retrievedSamples = stretcher.retrieve(outputChannels.data(), outputSize);

        size_t totalSamples = retrievedSamples * channels;
        float *output_data = static_cast<float *>(malloc(totalSamples * sizeof(float)));
        if (!output_data)
        {
            caml_failwith("Failed to allocate memory for output data");
        }

        if (channels > 1)
        {
            for (size_t i = 0; i < retrievedSamples; ++i)
                for (int c = 0; c < channels; ++c)
                    output_data[i * channels + c] = outputChannels[c][i];
        }
        else
        {
            std::copy(outputBuffers[0].begin(),
                      outputBuffers[0].begin() + retrievedSamples,
                      output_data);
        }

        intnat dims[channels];
        for (int c = 0; c < channels; ++c)
            dims[c] = static_cast<intnat>(totalSamples);

        v_output = caml_ba_alloc(CAML_BA_FLOAT32 | CAML_BA_C_LAYOUT, channels, output_data, dims);

        CAMLreturn(v_output);
    }
}
