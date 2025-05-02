#!/bin/bash

OUTPUT_DIR="$(pwd)/audio"

declare -A AUDIO_FORMATS=(
  ["wav"]="pcm_s16le"
  ["aif"]="pcm_s16be"
  ["flac"]="flac"
  ["mp3"]="libmp3lame"
  ["au"]="pcm_mulaw"
)

SAMPLE_RATES=(8000 22050 44100)
DURATION=1

declare -A SOURCES=(
    ["clipping"]="sine=f=440:sample_rate={SR}:duration={DUR},volume=volume=5"
    ["lsine"]="sine=f=10:sample_rate={SR}:duration={DUR}"
    ["stereo"]="sine=f=500:sample_rate={SR}:duration={DUR} [l]; aevalsrc=exprs='random(0)*0.5':duration={DUR}:sample_rate={SR} [r]; [l][r] amerge=inputs=2"
)

if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed." >&2
    exit 1
fi

total_files=$(( ${#SOURCES[@]} * ${#AUDIO_FORMATS[@]} * ${#SAMPLE_RATES[@]} ))
successful_files=0
current_file_number=0

for audio_source in "${!SOURCES[@]}"; do
    sft="${SOURCES[$audio_source]}"

    for format in "${!AUDIO_FORMATS[@]}"; do
        codec="${AUDIO_FORMATS[$format]}"

        for sr in "${SAMPLE_RATES[@]}"; do
            ((current_file_number++))

            source_key=$(echo "$audio_source" | tr -cd '[:alnum:]_-')
            filename="${format}_${source_key}_${sr}hz_${DURATION}s.${format}"
            output_path="$OUTPUT_DIR/$filename"

            source_filter="${sft//\{SR\}/$sr}"
            source_filter="${source_filter//\{DUR\}/$DURATION}"

            codec_opts=()
            case "$codec" in
                "libmp3lame") codec_opts+=("-b:a" "32k" "-compression_level" "9") ;;
                "libvorbis")  codec_opts+=("-q:a" "0") ;;
                "aac")        codec_opts+=("-b:a" "48k") ;;
            esac

            ffmpeg -y \
                -f lavfi -i "$source_filter" \
                -t "$DURATION" \
                -ar "$sr" \
                -c:a "$codec" "${codec_opts[@]}" \
                -vn \
                "$output_path" \
                -loglevel error > /dev/null 2>&1

            if [[ $? -eq 0 ]]; then
                ((successful_files++))
            fi
        done
    done
done

exit 0
