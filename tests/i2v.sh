#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1
PY_FILE=./generate.py
CKPT_DIR=/home/jialzhu/wan2.2model # set the path of model checkpoint

PROMPT="The camera opens with a medium shot of a young Korean woman with long dark hair, who looks like a modern idol, standing on a high-rise rooftop at night with city skyscrapers all around. She is wearing a soft knit top and silver-accented pants. At first, she holds a glowing, moon-like orb in her raised hand. The orb then gently dissolves, changing into a faint silver aurora that drifts from her palm. As the aurora spreads, a glowing silver moon chariot with two white horses appears and floats closely around her. The chariot shines with a soft lunar light, which reflects on the glass of the nearby buildings. The camera slowly zooms out, keeping her at the center, to show the full scene of her with the chariot and horses, making them all part of the wider city view. The scene is lit by the dreamy, soft light from the chariot against the dark, star-filled sky."

SAVEFILE_DIR=./generate_video

SAVEFILE_NAME=i2v-korean-woman-8gpu-seed493227.mp4
torchrun --nproc_per_node=8 $PY_FILE --task i2v-A14B --size 1280*720 --frame_num 121 --ckpt_dir $CKPT_DIR --prompt "$PROMPT" --image examples/korean_woman.jpeg --ulysses_size 8 --base_seed 493227 --save_file $SAVEFILE_DIR/$SAVEFILE_NAME
