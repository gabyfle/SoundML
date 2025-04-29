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

#include "read.hxx"
#include "write.hxx"

extern "C"
{
    CAMLprim value caml_read_audio_file_f32(value filename, value res_typ, value sample_rate, value fix)
    {
        return caml_read_audio_file<float>(filename, res_typ, sample_rate, fix);
    }

    CAMLprim value caml_read_audio_file_f64(value filename, value res_typ, value sample_rate, value fix)
    {
        return caml_read_audio_file<double>(filename, res_typ, sample_rate, fix);
    }

    CAMLprim value caml_write_audio_file_f32(value filename, value ba_data, value metadata)
    {
        return caml_write_audio_file<float>(filename, ba_data, metadata);
    }

    CAMLprim value caml_write_audio_file_f64(value filename, value ba_data, value metadata)
    {
        return caml_write_audio_file<double>(filename, ba_data, metadata);
    }
}
