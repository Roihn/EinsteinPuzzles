# Einstein Puzzles on Tabletop
<p align="center">
    <img src="doc/game_intro.png" width="400"/>
<p>
<p align="center">
          &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/Roihn/Einstein-Puzzles-Data">Hugging Face Dataset</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/Roihn/Einstein-Puzzles-Model">Hugging Face Model</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/xxxx.xxxxx">Paper</a>
</p>

**Communication and Verification in LLM Agents towards Collaboration under Information Asymmetry**

*Run Peng\*, Ziqiao Ma\*, Amy Pang, Sikai Li, Zhang Xi-Jia, Yingzhuo Yu, Cristian-Paul Bara, Joyce Chai*


## Environment Setup

We recommend using uv for the environment setup. 

```bash
uv sync
```

We fine-tuned and evaluated our models using NVIDIA A40 GPUs with 48GB memory and CUDA 12.4. Training was conducted on 4 GPUs, while evaluation used a single GPU. Please ensure your `torch` and `vllm` versions are compatible with this setup, and adjust accordingly if you encounter any issues.

## Training 

We provide training data for four action space configurations with Chain-of-Thought(CoT). We suggest you to download the dataset through our [huggingface](https://huggingface.co/datasets/Roihn/Einstein-Puzzles-Data), and store them under the `EinsteinPuzzles/dataset` folder. If you would like to have the version with no CoT, you can simply remove the contents inside `<think>` tags.

We use the fine-tuning framework from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) with version `0.6.0.post3` for all the fine-tunings. Key hyperparameters and settings used in training are documented in the appendix.

## Evaluation

We open-source our fine-tuned model on [llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) with both information providing and seeking capability and CoT reasoning on [huggingface](https://huggingface.co/Roihn/Einstein-Puzzles-Model). We recommend you to download it under the `EinsteinPuzzles/checkpoint` folder. More checkpoints are available through request.

For evaluation, we include 300 unseen test cases for online interaction, stored in `dataset/eval/eval_game_ids.json`. Each test case is indexed by a unique game ID that specifies the initial state of the game.

Once the checkpoint is available, you can evaluate the model (without verifier) using the following command:
```bash
cd Einstein_Puzzles
CUDA_VISIBLE_DEVICES=0 uv run src/eval/eval_game_raw_model.py \
--action_mode provide_seek \
--use_cot \
--output_dir outputs/ \
--max_files 300 \
--json_path dataset/eval/eval_game_ids.json \
--base_model_path meta-llama/Llama-3.1-8B-Instruct \
--lora_model_path ./checkpoint/llama3.1-8B-cot-provide-seek
```

For the evaluation with verifier, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 uv run src/eval/eval_game_verifier_model.py \
--use_cot \
--action_mode provide_seek \
--output_dir outputs_verifier/ \
--max_files 300 \
--json_path dataset/eval/eval_game_ids.json \
--base_model_path meta-llama/Llama-3.1-8B-Instruct \
--lora_model_path ./checkpoint/llama3.1-8B-cot-provide-seek \
--verifier <affordance_verifier/communication_verifier/reasoning_verifier>
```

## Citation

```bibtex
```