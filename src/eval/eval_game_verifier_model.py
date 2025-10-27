from eval_base import *
from itertools import permutations
from random import seed, shuffle, choice
import requests
from copy import deepcopy
from time import time
from tqdm import tqdm
import json
import os
import argparse


def process_single_pkl(model, pkl_path, action_mode, is_whole_dialog, use_cot, output_dir, allow_random_guess=True, verifier="reasoning_verifier"):
    all_blocks = [o for o in Block]
    # Set random seed
    seed(0)
    
    # Extract just the filename from the path
    filename = os.path.basename(pkl_path)
    file_id = os.path.splitext(filename)[0]  # Remove extension
    
    print(f"\nProcessing {filename}...")
    
    # Parse configuration from filename
    num_obj, starting_bins_choice, final_locations_choice, rel_ids, knowledge_disparity = parse_and_reconstruct_config(pkl_path)
    
    # # Skip files that don't have exactly 4 objects
    # if num_obj != 4:
    #     print(f"Skipping {filename} - does not have exactly 4 objects")
    #     return {
    #         "file_id": file_id,
    #         "status": "skipped",
    #         "reason": "not 4 objects",
    #         "objects": num_obj
    #     }
        
    blocks = all_blocks[:num_obj]

    final_locations = {blk: loc for blk, loc in zip(blocks, final_locations_choice)}
    starting_locations = {block: choice_loc for block, choice_loc in zip(blocks, starting_bins_choice)}
    grounding = Grounding(blocks[0], final_locations[blocks[0]])
    relations = []
    
    for i in range(1, len(final_locations)):
        obj = blocks[i]
        anchor = blocks[rel_ids[i]]
        # If x.value == y.value, then they share the same row/column
        # Taking 1 - int(x.value == y.value) will give 1 if they have different row/column; 0 otherwise
        rel = RelationType(sum([(1-int(x.value==y.value))*2**(1-j) for j, (x, y) in enumerate(zip(final_locations[obj].value, final_locations[anchor].value))]))
        relations.append(Relation(obj, anchor, rel, RelationValue.SAME))

    state = State(starting_locations)
    relations1 = [x for x, d in zip(relations, knowledge_disparity) if d <= 1]
    relations2 = [x for x, d in zip(relations, knowledge_disparity) if d >= 1]

    knowledge1 = Knowledge(relations1, [grounding])
    knowledge2 = Knowledge(relations2, [grounding])

    env = EinsteinGameWithVerifier(
        init_state=state,
        knowledge1=knowledge1,
        knowledge2=knowledge2,
        final_state=State(final_locations),
        is_whole_dialog=is_whole_dialog,
        action_mode=action_mode,
        num_blocks=num_obj,
        use_cot=use_cot,
        allow_random_guess=allow_random_guess,
        verifier=verifier)
    
    print("#"*60)
    print(f"File ID: {file_id}")
    print(state.for_vis())
    print(knowledge1)
    print(knowledge2)
    print("#"*60)
    
    results = {
        "file_id": file_id,
        "pkl_path": pkl_path,
        "num_objects": num_obj,
        "step_count": 0,
        "terminated": False,
        "truncated": False,
        "dialog": []
    }
    
    # Track used trials and sum_err for this game
    total_used_trials = 0
    total_sum_err = 0
    total_corrected_count = 0
    
    # try:
    while True:
        dialog = env.render()
        # print("#"*60, f"dialog {env.cur_player.perspective.value}", "#"*60)
        # print(json.dumps(dialog, indent=4))
        # print("#"*60, f"dialog {env.cur_player.perspective.value}", "#"*60)
         
        action_list = model.predict_n(dialog, max_tokens=192, temperature=0.2)
        
        # print("#"*60, f"response list {env.cur_player.perspective.value}", "#"*60)
        # print(action_list)
        # print("#"*60, f"response {env.cur_player.perspective.value}", "#"*60)
        
        # Store dialog and response
        results["dialog"].append({
            "player": env.cur_player.perspective.value,
            "dialog": deepcopy(dialog),
            "response": action_list,
            "step": env.step_count
        })
        
        state, reward, terminated, truncated, info = env.step(action_list)
        
        results["dialog"][-1]["used_trials"] = info["used_trials"]
        results["dialog"][-1]["sum_err"] = info["sum_err"]
        results["dialog"][-1]["actions_err"] = info["actions_err"]
        results["dialog"][-1]["corrected"] = info["corrected"]
        
        # Track metrics
        total_used_trials += info["used_trials"]
        total_sum_err += info["sum_err"]
        total_corrected_count += 1 if info["corrected"] else 0

        if terminated or truncated:
            break
    
    results["step_count"] = env.step_count
    results["terminated"] = terminated
    results["truncated"] = truncated
    results["success"] = terminated and not truncated
    num_correct, _ = partial_success_rate(env.final_state, state)
    
    results["partial_success_rate"] = num_correct
    results["num_obj"] = num_obj
    
    # Add new metrics to results
    results["total_used_trials"] = total_used_trials
    results["avg_used_trials"] = total_used_trials / max(1, env.step_count)
    results["total_sum_err"] = total_sum_err
    results["avg_sum_err"] = total_sum_err / max(1, env.step_count)
    results["avg_corrected"] = total_corrected_count / max(1, env.step_count)
    
    # if terminated:
    #     print("#"*60, f"terminated, step_count: {env.step_count}", "#"*60)
    # if truncated:
    #     print("#"*60, "truncated", "#"*60)
    
    # except Exception as e:
    #     print(f"Error processing {file_id}: {str(e)}")
    #     results["error"] = str(e)
    #     results["success"] = False
    
    if not os.path.exists(os.path.join(output_dir, "json")):
        os.makedirs(os.path.join(output_dir, "json"))
        
    # Save the full dialog results to a separate file
    dialog_file = os.path.join(output_dir, f"json/dialog_{file_id}.json")
    with open(dialog_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Return just the metrics for the summary file
    return {
        "file_id": file_id,
        "pkl_path": pkl_path,
        "num_objects": num_obj,
        "step_count": results["step_count"],
        "terminated": results["terminated"],
        "truncated": results["truncated"],
        "partial_success_rate": results["partial_success_rate"],
        "success": results.get("success", False),
        "error": results.get("error", None),
        "total_used_trials": total_used_trials,
        "avg_used_trials": results["avg_used_trials"],
        "total_sum_err": total_sum_err,
        "avg_sum_err": results["avg_sum_err"],
        "avg_corrected": results["avg_corrected"],
    }

def main():
    parser.add_argument("--json_path", type=str, 
                        default="dataset/eval/sampled_solutions_provide_seek_1000.json",
                        help="Path to JSON file containing list of PKL files")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--max_files", type=int, default=500,
                        help="Maximum number of files to process (for testing)")
    parser.add_argument("--max_files_per_num_obj", type=int, default=None)
    parser.add_argument("--allow_random_guess", default=True)
    parser.add_argument("--verifier", type=str, default="reasoning_verifier",)
    args = parser.parse_args()
    
    # set the random seed
    seed(42)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the JSON file containing the list of PKL files
    with open(args.json_path, 'r') as f:
        pkl_files = json.load(f)
    
    if args.max_files_per_num_obj is not None:
        # Group files by number of objects
        files_by_num_obj = {}
        for pkl_path in pkl_files:
            # Extract num_obj from the path or filename
            filename = os.path.basename(pkl_path)
            # Parse configuration from filename to get num_obj
            num_obj, _, _, _, _ = parse_and_reconstruct_config(pkl_path)
            
            if num_obj not in files_by_num_obj:
                files_by_num_obj[num_obj] = []
            files_by_num_obj[num_obj].append(pkl_path)
        
        # Sample max_files_per_num_obj from each group
        sampled_files = []
        for num_obj, file_list in files_by_num_obj.items():
            # Shuffle files for fair sampling
            shuffle(file_list)
            # Take up to max_files_per_num_obj
            sampled_files.extend(file_list[:args.max_files_per_num_obj])
        
        # Update pkl_files with the sampled subset
        pkl_files = sampled_files
        print(f"Sampled {len(pkl_files)} files evenly across object counts (up to {args.max_files_per_num_obj} per count)")
    elif args.max_files is not None:
        # Just take the first max_files if no per_num_obj sampling
        pkl_files = pkl_files[:args.max_files]
        print(f"Using first {len(pkl_files)} files from the input list")
    
    all_metrics = []
    
    if "llama" in args.base_model_path.lower():
        from local_model import LoRAChatLLaMAModel as LoRAChatModel
    elif "qwen" in args.base_model_path.lower():
        from local_model import LoRAChatQwenModel as LoRAChatModel
    
    # from local_model import LoRAChatLLaMAModel as LoRAChatModel
    model = LoRAChatModel(
        base_model_name=args.base_model_path,
        lora_path=args.lora_model_path
    )

    # Process each PKL file
    for pkl_path in tqdm(pkl_files, desc="Processing PKL files"):
        metrics = process_single_pkl(
            model=model,
            pkl_path=pkl_path,
            action_mode=args.action_mode,
            is_whole_dialog=args.is_whole_dialog,
            use_cot=args.use_cot,
            output_dir=args.output_dir,
            allow_random_guess=args.allow_random_guess,
            verifier=args.verifier
        )
        all_metrics.append(metrics)
    
    # Calculate summary statistics
    total_files = len(all_metrics)
    successful_files = sum(1 for m in all_metrics if m.get("success", False))
    success_rate = successful_files / total_files if total_files > 0 else 0
    
    # Calculate average steps for successful runs
    successful_steps = [m["step_count"] for m in all_metrics if m.get("success", False)]
    avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else 0
    
    # Group by number of objects
    by_objects = {}
    for m in all_metrics:
        num_obj = m.get("num_objects", 0)
        if num_obj not in by_objects:
            by_objects[num_obj] = {"total": 0, "success": 0}
        by_objects[num_obj]["total"] += 1
        if m.get("success", False):
            by_objects[num_obj]["success"] += 1
    
    # Calculate success rate by number of objects
    for obj_count in by_objects:
        if by_objects[obj_count]["total"] > 0:
            by_objects[obj_count]["success_rate"] = by_objects[obj_count]["success"] / by_objects[obj_count]["total"]
        else:
            by_objects[obj_count]["success_rate"] = 0
    
    # Calculate partial success rate by number of objects
    by_objects_partial = {}
    for m in all_metrics:
        num_obj = m.get("num_objects", 0)
        if num_obj not in by_objects_partial:
            by_objects_partial[num_obj] = {"total": 0, "partial_success_sum": 0, "normalized_sum": 0}
        
        by_objects_partial[num_obj]["total"] += 1
        
        # Get the partial success rate for this file
        partial_success = m.get("partial_success_rate", 0)
        by_objects_partial[num_obj]["partial_success_sum"] += partial_success
        
        # Calculate normalized success rate (0-1 scale)
        normalized_rate = partial_success / num_obj if num_obj > 0 else 0
        by_objects_partial[num_obj]["normalized_sum"] += normalized_rate

    # Calculate average partial success rate by number of objects
    for obj_count in by_objects_partial:
        if by_objects_partial[obj_count]["total"] > 0:
            # Raw average (actual number of correct objects)
            by_objects_partial[obj_count]["avg_partial_success"] = by_objects_partial[obj_count]["partial_success_sum"] / by_objects_partial[obj_count]["total"]
            
            # Normalized average (0-1 scale)
            by_objects_partial[obj_count]["avg_normalized_success"] = by_objects_partial[obj_count]["normalized_sum"] / by_objects_partial[obj_count]["total"]
        else:
            by_objects_partial[obj_count]["avg_partial_success"] = 0
            by_objects_partial[obj_count]["avg_normalized_success"] = 0

    # Calculate overall average normalized partial success rate
    normalized_sum = sum(m.get("partial_success_rate", 0) / m.get("num_objects", 1) for m in all_metrics)
    avg_normalized_partial_success = normalized_sum / total_files if total_files > 0 else 0
    
    # NEW: Calculate metrics for used trials and sum errors
    # Group by number of objects for used trials
    by_objects_trials = {}
    for m in all_metrics:
        num_obj = m.get("num_objects", 0)
        if num_obj not in by_objects_trials:
            by_objects_trials[num_obj] = {"total": 0, "avg_used_trials_sum": 0}
        
        by_objects_trials[num_obj]["total"] += 1
        by_objects_trials[num_obj]["avg_used_trials_sum"] += m.get("avg_used_trials", 0)
    
    # Group by number of objects for sum errors
    by_objects_errors = {}
    for m in all_metrics:
        num_obj = m.get("num_objects", 0)
        if num_obj not in by_objects_errors:
            by_objects_errors[num_obj] = {"total": 0, "total_sum_err_sum": 0}
        
        by_objects_errors[num_obj]["total"] += 1
        by_objects_errors[num_obj]["total_sum_err_sum"] += m.get("total_sum_err", 0)
    
    # Calculate average used trials and errors by number of objects
    for obj_count in by_objects_trials:
        if by_objects_trials[obj_count]["total"] > 0:
            by_objects_trials[obj_count]["avg_used_trials"] = by_objects_trials[obj_count]["avg_used_trials_sum"] / by_objects_trials[obj_count]["total"]
        else:
            by_objects_trials[obj_count]["avg_used_trials"] = 0
    
    for obj_count in by_objects_errors:
        if by_objects_errors[obj_count]["total"] > 0:
            by_objects_errors[obj_count]["avg_total_sum_err"] = by_objects_errors[obj_count]["total_sum_err_sum"] / by_objects_errors[obj_count]["total"]
        else:
            by_objects_errors[obj_count]["avg_total_sum_err"] = 0
    
    # Calculate overall averages
    total_avg_used_trials_sum = sum(m.get("avg_used_trials", 0) for m in all_metrics)
    overall_avg_used_trials = total_avg_used_trials_sum / total_files if total_files > 0 else 0
    
    total_sum_err_sum = sum(m.get("total_sum_err", 0) for m in all_metrics)
    overall_avg_total_sum_err = total_sum_err_sum / total_files if total_files > 0 else 0
    
    # Prepare summary
    summary = {
        "total_files": total_files,
        "successful_files": successful_files,
        "success_rate": success_rate,
        "average_steps_for_success": avg_steps,
        "by_objects": by_objects,
        "by_objects_partial": by_objects_partial,
        "avg_normalized_partial_success": avg_normalized_partial_success,
        "by_objects_trials": by_objects_trials,
        "by_objects_errors": by_objects_errors,
        "overall_avg_used_trials": overall_avg_used_trials,
        "overall_avg_total_sum_err": overall_avg_total_sum_err,
        "timestamp": time()
    }
    
    # Save metrics and summary
    metrics_file = os.path.join(args.output_dir, "all_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print summary to console
    print("\n" + "="*80)
    print(f"Evaluation Complete - Results in {args.output_dir}")
    print(f"Total files processed: {total_files}")
    print(f"Success rate: {success_rate:.2%} ({successful_files}/{total_files})")
    print(f"Average steps for successful runs: {avg_steps:.2f}")
    print(f"Average normalized partial success rate: {avg_normalized_partial_success:.2f}")
    print(f"Overall average used trials per step: {overall_avg_used_trials:.2f}")
    print(f"Overall average sum of errors per game: {overall_avg_total_sum_err:.2f}")
    print(f"Overall average correct time: {sum(m.get('avg_corrected', 0) for m in all_metrics) / total_files if total_files > 0 else 0:.2f}")
    print("Success rate by number of objects:")
    for obj_count in sorted(by_objects.keys()):
        obj_data = by_objects[obj_count]
        print(f"  {obj_count} objects: {obj_data['success_rate']:.2%} ({obj_data['success']}/{obj_data['total']})")
    print("Partial success rate by number of objects:")
    for obj_count in sorted(by_objects_partial.keys()):
        obj_data = by_objects_partial[obj_count]
        print(f"  {obj_count} objects: {obj_data['avg_partial_success']:.2f}")
    print("Average used trials per step by number of objects:")
    for obj_count in sorted(by_objects_trials.keys()):
        obj_data = by_objects_trials[obj_count]
        print(f"  {obj_count} objects: {obj_data['avg_used_trials']:.2f}")
    print("Average sum of errors per game by number of objects:")
    for obj_count in sorted(by_objects_errors.keys()):
        obj_data = by_objects_errors[obj_count]
        print(f"  {obj_count} objects: {obj_data['avg_total_sum_err']:.2f}")
    print("="*80)
    print("Average correct time by number of objects:")
    by_objects_correct_time = {}
    for m in all_metrics:
        num_obj = m.get("num_objects", 0)
        if num_obj not in by_objects_correct_time:
            by_objects_correct_time[num_obj] = {"total": 0, "correct_time_sum": 0}
        
        by_objects_correct_time[num_obj]["total"] += 1
        by_objects_correct_time[num_obj]["correct_time_sum"] += m.get("avg_corrected", 0)

    for obj_count in sorted(by_objects_correct_time.keys()):
        if by_objects_correct_time[obj_count]["total"] > 0:
            avg = by_objects_correct_time[obj_count]["correct_time_sum"] / by_objects_correct_time[obj_count]["total"]
            print(f"  {obj_count} objects: {avg:.2f}")

    
    # print summary to local "result.txt" file
    with open(os.path.join(args.output_dir, "result.txt"), 'w') as f:
        f.write(f"Evaluation Complete - Results in {args.output_dir}\n")
        f.write(f"Total files processed: {total_files}\n")
        f.write(f"Success rate: {success_rate:.2%} ({successful_files}/{total_files})\n")
        f.write(f"Average steps for successful runs: {avg_steps:.2f}\n")
        f.write(f"Average normalized partial success rate: {avg_normalized_partial_success:.2f}\n")
        f.write(f"Overall average used trials per step: {overall_avg_used_trials:.2f}\n")
        f.write(f"Overall average sum of errors per game: {overall_avg_total_sum_err:.2f}\n")
        f.write(f"Overall average correct time: {sum(m.get('avg_corrected', 0) for m in all_metrics) / total_files if total_files > 0 else 0:.2f}\n")
        f.write("Success rate by number of objects:\n")
        for obj_count in sorted(by_objects.keys()):
            obj_data = by_objects[obj_count]
            f.write(f"  {obj_count} objects: {obj_data['success_rate']:.2%} ({obj_data['success']}/{obj_data['total']})\n")
        f.write("Partial success rate by number of objects:\n")
        for obj_count in sorted(by_objects_partial.keys()):
            obj_data = by_objects_partial[obj_count]
            f.write(f"  {obj_count} objects: {obj_data['avg_partial_success']:.2f}\n")
        f.write("Average used trials per step by number of objects:\n")
        for obj_count in sorted(by_objects_trials.keys()):
            obj_data = by_objects_trials[obj_count]
            f.write(f"  {obj_count} objects: {obj_data['avg_used_trials']:.2f}\n")
        f.write("Average sum of errors per game by number of objects:\n")
        for obj_count in sorted(by_objects_errors.keys()):
            obj_data = by_objects_errors[obj_count]
            f.write(f"  {obj_count} objects: {obj_data['avg_total_sum_err']:.2f}\n")
        f.write("Average correct time by number of objects:\n")
        for obj_count in sorted(by_objects_correct_time.keys()):
            if by_objects_correct_time[obj_count]["total"] > 0:
                avg = by_objects_correct_time[obj_count]["correct_time_sum"] / by_objects_correct_time[obj_count]["total"]
                f.write(f"  {obj_count} objects: {avg:.2f}\n")


if __name__ == "__main__":
    main()