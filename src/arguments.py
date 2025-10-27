import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--action_mode", type=str, default="provide_seek")
parser.add_argument("--num_games", type=int, default=1)
parser.add_argument("--max_num_objects", type=int, default=4)
parser.add_argument("--use_cot", action="store_true")
parser.add_argument("--is_whole_dialog", action="store_true")
parser.add_argument("--base_model_path", type=str, default="pre-trained-weights/Meta-Llama-3-8B-Instruct-HF")
parser.add_argument("--lora_model_path", type=str, default="checkpoint/llama3-8b-sft-share-ask/")
parser.add_argument(
        '--id',
        type=str,
        default="6_220002_312003_000221_12022_00",
        help="Specifies the game ID (default: 6_220002_312003_000221_12022_00)."
    )
