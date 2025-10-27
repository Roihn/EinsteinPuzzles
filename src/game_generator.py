import argparse
import random
from copy import deepcopy
import os, pickle
from itertools import permutations
import heapq
from time import time
from tqdm import tqdm
import glob
import re

from environment import Block, Bins, Grounding, Move, StateMoveAndKnowledge, RelationValue, RelationType, Knowledge, Perspective, Relation, State, Player

class AutoCounter:
    def __init__(self, start=0):
        self.count = start
    
    def __call__(self):
        current = self.count
        self.count += 1
        return current


def str_hash(x):
    return int.from_bytes(str(x).encode(), 'little')


def evaluate_game(players, state, final_locations, action_mode="provide_seek", max_steps=2000, max_sol_num=5):
    if not players[0].get_moves(state)[0]:
        players = list(reversed(players))
    
    complete_games = []
    interrupted_games = []
    failed_games = []
    known_states = set()
    known_state_saves = list()
    complete_states = set([str_hash(str(state))])
    step = -1
    counter = AutoCounter()
    initial_tuple = ([state], players, [], [], [])
    working_queue = [(len(initial_tuple[0]), counter(), initial_tuple)]
    heapq.heapify(working_queue)
    
    while working_queue:
        step += 1
        _, _, (prev_states, current_players, move_list, prev_available_move, prev_players) = heapq.heappop(working_queue)
        prev_state = prev_states[-1]
        
        if len(complete_games) >= max_sol_num:
            break
        if step > max_steps:
            break
        if prev_state.is_final(State(final_locations)):
            complete_games.append((prev_states, current_players, move_list, prev_available_move, prev_players))
            for st, persp in zip(prev_states, prev_players):
                complete_states.add(str_hash(str(st)+str(persp)))
                
            interrupted_games = [x for x in interrupted_games if not all(str_hash(str(s) + str(p)) in complete_states for s,p in zip(x[0],x[-1]))]
            continue
        
        if len(move_list) > 1:
            if all(x is None for x in move_list[-4:]):
                failed_games.append((prev_states, current_players, move_list, prev_available_move, prev_players))
                continue
            if len(move_list) > 100:
                failed_games.append((prev_states, current_players, move_list, prev_available_move, prev_players))
                continue
        
        player, partner = current_players
        moves, share_dlg, ask_dlg = player.get_moves(prev_state)
        
        if action_mode == "provide_seek":
            all_moves = moves + share_dlg + ask_dlg
        elif action_mode == "provide":
            all_moves = moves + share_dlg
        elif action_mode == "seek":
            all_moves = moves + ask_dlg
        else:
            all_moves = moves
        has_updated = False
        if all_moves:
            for chosen_move in all_moves:
                if isinstance(chosen_move, Move):
                    player = deepcopy(current_players[0])
                    partner = deepcopy(current_players[1])
                    new_state = player.make_move(prev_state, chosen_move, State(final_locations), partner)                        
                    if player.perspective == Perspective.PLAYER_1:
                        new_state_knowledge = StateMoveAndKnowledge(new_state, player.get_infered_knowledge(), partner.get_infered_knowledge(), chosen_move)
                    else:
                        new_state_knowledge = StateMoveAndKnowledge(new_state, partner.get_infered_knowledge(), player.get_infered_knowledge(), chosen_move)
                    if str_hash(str(new_state_knowledge)) in known_states:
                        interrupted_games.append((prev_states + [new_state], future_players, move_list+[chosen_move], prev_available_move+[all_moves], prev_players+[deepcopy(player)]))
                        continue
                    else:
                        known_state_saves.append((prev_states + [new_state], [deepcopy(partner), deepcopy(player)], move_list+[chosen_move], prev_available_move+[all_moves], prev_players+[deepcopy(player)]))
                        known_states.add(str_hash(str(new_state_knowledge)))
                    future_players = [partner, player]
                    new_tuple = (prev_states + [new_state], future_players, move_list+[chosen_move], prev_available_move+[all_moves], prev_players+[deepcopy(player)])
                    heapq.heappush(working_queue, (len(new_tuple[0]), counter(), new_tuple))
                    has_updated = True
                elif isinstance(chosen_move, Block):
                    # ask for block's knowledge
                    # only pick the answer from the initially given grounding and relations
                    shareable_knowledge = []
                    partner_knowledge = partner.knowledge
                    for grounding in partner_knowledge.groundings:
                        if grounding.block == chosen_move:
                            shareable_knowledge.append(grounding)
                    
                    for relation in partner_knowledge.relations:
                        if relation.blocks[0] == chosen_move or relation.blocks[1] == chosen_move and partner.shared_knowledge_flags[relation] == False:
                            shareable_knowledge.append(relation)
                    if len(shareable_knowledge) == 0:
                        continue
                    for knowledge in shareable_knowledge:
                        # When ask for knowledge, there will be multiple possible futures that can be generated
                        player = deepcopy(current_players[0])
                        partner = deepcopy(current_players[1])
                        new_state = deepcopy(prev_state)
                        partner.share_knowledge(knowledge, player)
                        future_players = [player, partner]
                        new_tuple = (prev_states + [new_state] * 2, future_players, move_list+[chosen_move]+[knowledge], prev_available_move+[all_moves], prev_players+[deepcopy(player), deepcopy(partner)])
                        heapq.heappush(working_queue, (len(new_tuple[0]), counter(), new_tuple))
                        has_updated = True
                else:
                    player = deepcopy(current_players[0])
                    partner = deepcopy(current_players[1])
                    # share relation
                    if not chosen_move in partner.knowledge.relations and not chosen_move in partner.knowledge.groundings:
                        new_state = deepcopy(prev_state)
                        player.share_knowledge(chosen_move, partner)
                    else:
                        continue
                    future_players = [partner, player]
                    new_tuple = (prev_states + [new_state], future_players, move_list+[chosen_move], prev_available_move+[all_moves], prev_players+[deepcopy(player)])
                    heapq.heappush(working_queue, (len(new_tuple[0]), counter(), new_tuple))
                    has_updated = True
        else:
            player = current_players[0]
            partner = current_players[1]
            chosen_move = None
            new_state = deepcopy(prev_state)
            future_players = [deepcopy(partner), deepcopy(player)]
            new_tuple = (prev_states + [new_state], future_players, move_list+[chosen_move], prev_available_move+[all_moves], prev_players+[deepcopy(player)])
            heapq.heappush(working_queue, (len(new_tuple[0]), counter(), new_tuple))
            has_updated = True
        
        if action_mode == "none":
            available_moves = player.get_available_moves(prev_state)
            if available_moves:
                for chosen_move in available_moves:
                    player = deepcopy(current_players[0])
                    partner = deepcopy(current_players[1])
                    new_state = player.make_move(state=prev_state, move=chosen_move, final_state=State(final_locations), partner=partner)
                    if player.perspective == Perspective.PLAYER_1:
                        new_state_knowledge = StateMoveAndKnowledge(new_state, player.get_infered_knowledge(), partner.get_infered_knowledge(), chosen_move)
                    else:
                        new_state_knowledge = StateMoveAndKnowledge(new_state, partner.get_infered_knowledge(), player.get_infered_knowledge(), chosen_move)
                    
                    if str_hash(str(new_state_knowledge)) in known_states:
                        interrupted_games.append((prev_states + [new_state], future_players, move_list+[chosen_move], prev_available_move+[all_moves], prev_players+[deepcopy(player)]))
                        continue
                    else:
                        known_state_saves.append((prev_states + [new_state], [deepcopy(partner), deepcopy(player)], move_list+[chosen_move], prev_available_move+[all_moves], prev_players+[deepcopy(player)]))
                        known_states.add(str_hash(str(new_state_knowledge)))
                    future_players = [partner, player]
                    new_tuple = (prev_states + [new_state], future_players, move_list+[chosen_move], prev_available_move+[all_moves], prev_players+[deepcopy(player)])
                    heapq.heappush(working_queue, (len(new_tuple[0]), counter(), new_tuple))
                    has_updated = True
        
        if not has_updated:
            # pass the turn
            player = deepcopy(current_players[0])
            partner = deepcopy(current_players[1])
            new_state = deepcopy(prev_state)
            future_players = [partner, player]
            new_tuple = (prev_states + [new_state], future_players, move_list+[None], prev_available_move+[all_moves], prev_players+[deepcopy(player)])
            heapq.heappush(working_queue, (len(new_tuple[0]), counter(), new_tuple))
            has_updated = True

    complete_games = sorted(complete_games, key=lambda x: len(x[0]))          
    interrupted_games = sorted(interrupted_games, key=lambda x: len(x[0]))    
    finished_games = []
    
    return complete_games, failed_games, interrupted_games, known_states, finished_games


def generate_config_id(num_obj, starting_bins_choice, final_locations_choice, rel_ids, knowledge_disparity):
    """Generate a unique config ID based on the given parameters"""
    config_id = f"{str(num_obj)}_{''.join([str(x.value) for x in starting_bins_choice])}_{''.join([str(x.value[0].value+2*x.value[1].value) for x in final_locations_choice])}_{''.join([str(x) for x in rel_ids])}_{''.join([str(x) for x in knowledge_disparity])}"
    return config_id


def config_exists(config_id, output_dir):
    """Check if a config already exists in the output directory"""
    pattern = os.path.join(output_dir, f"{config_id}_*.pkl")
    existing_files = glob.glob(pattern)
    return len(existing_files) > 0


def count_existing_configs(output_dir, num_obj):
    """Count the number of unique configs in the output directory for a specific number of objects"""
    # Get all pkl files
    all_files = glob.glob(os.path.join(output_dir, "*.pkl"))
    
    # Extract unique config IDs with the specified number of objects
    unique_configs = set()
    pattern = re.compile(f"^{num_obj}_")  # Match configs starting with num_obj_
    
    for file in all_files:
        # Extract base filename without path and extension
        filename = os.path.basename(file)
        # Remove solution number suffix (_XX.pkl)
        config_id = "_".join(filename.split("_")[:-1])
        
        # Only count configs with the matching number of objects
        if pattern.match(config_id):
            unique_configs.add(config_id)
    
    return len(unique_configs)


def sample_unique_config(num_obj, starting_bins_choices, final_locations_choices, rel_idx_lst, knowledge_disparity_lst, output_dir):
    """Sample a configuration that doesn't already exist in the output directory"""
    max_attempts = 100  # Prevent infinite loop
    
    for _ in range(max_attempts):
        # Randomly sample indices
        starting_bin_idx = random.randint(0, len(starting_bins_choices) - 1)
        final_location_idx = random.randint(0, len(final_locations_choices) - 1)
        rel_idx_idx = random.randint(0, len(rel_idx_lst) - 1)
        knowledge_disparity_idx = random.randint(0, len(knowledge_disparity_lst) - 1)
        
        # Get actual values
        starting_bins_choice = starting_bins_choices[starting_bin_idx]
        final_locations_choice = final_locations_choices[final_location_idx]
        rel_ids = rel_idx_lst[rel_idx_idx]
        knowledge_disparity = knowledge_disparity_lst[knowledge_disparity_idx]
        
        # Generate config ID
        config_id = generate_config_id(num_obj, starting_bins_choice, final_locations_choice, rel_ids, knowledge_disparity)
        
        # Check if this config already exists
        if not config_exists(config_id, output_dir):
            # Return the configuration if it doesn't exist
            return (num_obj, starting_bins_choice, final_locations_choice, rel_ids, knowledge_disparity)
    
    # If we couldn't find a unique config after max_attempts
    return None


def main(args):
    action_mode = args.action_mode
    num_games = args.num_games
    num_obj = args.max_num_objects
    verbose = args.verbose
    max_sol_num = args.max_sol_num
    all_blocks = [o for o in Block]
    starting_bins = [Bins.PLAYER_1_BIN, Bins.PLAYER_2_BIN]
    destination_bins = [Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN, Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN]
    
    # set random seed
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    blocks = all_blocks[:num_obj]
    starting_bins_choices = []
    for x in permutations(starting_bins*(num_obj//2+1)):
        if x[:num_obj] not in starting_bins_choices:
            starting_bins_choices.append(x[:num_obj])
            
    if num_obj % 2:
        # It should be either 1 more or 1 less than the half of the number of objects for each player
        starting_bins_choices = [x for x in starting_bins_choices if sum([y.value for y in x]) in [num_obj-1, num_obj+1]]
    else:
        # The objects need to be evenly distributed
        starting_bins_choices = [x for x in starting_bins_choices if sum([y.value for y in x]) == num_obj]

    # The final locations always guarantee that each bin is covered
    final_locations_choices = []
    for x in permutations(destination_bins*(num_obj//4+1)):
        if (Bins.TOP_LEFT_BIN in x[:num_obj]) and (Bins.TOP_RIGHT_BIN in x[:num_obj]) and (Bins.BOTTOM_LEFT_BIN in x[:num_obj]) and (Bins.BOTTOM_RIGHT_BIN in x[:num_obj]) and x[:num_obj] not in final_locations_choices:
            final_locations_choices.append(x[:num_obj])

    rel_idx_lst = []
    working_queue = [[0]]
    while working_queue:
        x = working_queue.pop(0)
        if len(x) == num_obj:
            rel_idx_lst.append(x)
        else:
            l = len(x)
            for i in range(l):
                working_queue.append(x+[i])

    # Knowledge disparity: 0 means only known by player 1; 1 means shared knowledge; 2 means only known by player 2
    knowledge_disparity_lst = []
    for x in permutations([0,1,2]*(num_obj//3+1)):
        if (0 in x[:num_obj-1]) and (1 in x[:num_obj-1]) and (2 in x[:num_obj-1]) and x[:num_obj-1] not in knowledge_disparity_lst:
            knowledge_disparity_lst.append(x[:num_obj-1])

    count = 0
    count_failed = 0
    
    # Keep generating configs until we reach num_games or run out of unique possibilities
    print(f"Target number of games: {num_games}")
    
    while True:
        # Check if we've already reached the target number of games for the current num_obj
        existing_configs_count = count_existing_configs(output_dir, num_obj)
        if existing_configs_count >= num_games:
            print(f"Target reached: {existing_configs_count}/{num_games} configs exist for {num_obj} objects")
            break
        
        print(f"Current progress: {existing_configs_count}/{num_games} configs for {num_obj} objects")
        
        # Sample a unique configuration
        config = sample_unique_config(num_obj, starting_bins_choices, final_locations_choices, 
                                     rel_idx_lst, knowledge_disparity_lst, output_dir)
        
        if config is None:
            print(f"Could not find any more unique configurations for {num_obj} objects. Exiting.")
            break
        
        num_obj, starting_bins_choice, final_locations_choice, rel_ids, knowledge_disparity = config
        
        # Generate config ID for tracking
        config_id = generate_config_id(num_obj, starting_bins_choice, final_locations_choice, rel_ids, knowledge_disparity)
        out_file_name = os.path.join(output_dir, config_id)
        
        print(f"Processing config: {config_id}")
        count += 1

        final_locations = {blk: loc for blk, loc in zip(blocks, final_locations_choice)}
        starting_locations = {block: choice_loc for block, choice_loc in zip(blocks, starting_bins_choice)}
        grounding = Grounding(blocks[0], final_locations[blocks[0]])
        relations = []
        for i in range(1, len(final_locations)):
            obj = blocks[i]
            anchor = blocks[rel_ids[i]]
            rel = RelationType(sum([(1-int(x.value==y.value))*2**(1-j) for j, (x, y) in enumerate(zip(final_locations[obj].value, final_locations[anchor].value))]))
            relations.append(Relation(obj, anchor, rel, RelationValue.SAME))

        state = State(starting_locations)
        relations1 = [x for x, d in zip(relations, knowledge_disparity) if d <= 1]
        relations2 = [x for x, d in zip(relations, knowledge_disparity) if d >= 1]

        knowledge1 = Knowledge(relations1, [grounding])
        knowledge2 = Knowledge(relations2, [grounding])

        player1 = Player(Perspective.PLAYER_1, knowledge1, num_blocks=num_obj, shared_knowledge_flags=None)
        player2 = Player(Perspective.PLAYER_2, knowledge2, num_blocks=num_obj, shared_knowledge_flags=None)
        players = player1, player2

        complete_games, failed_games, interrupted_games, known_states, finished_games = evaluate_game(
            players, state, final_locations, action_mode=action_mode, max_steps=args.max_steps, max_sol_num=max_sol_num
        )
        
        if complete_games:
            for i in range(len(complete_games)):
                res = {
                    "states": complete_games[i][0],
                    "final_locations": final_locations,
                    "knowledge1": knowledge1,
                    "knowledge2": knowledge2,
                    "moves": complete_games[i][2],
                    "first_player": complete_games[i][1][0].perspective,
                }
                
                if not verbose:
                    with open(out_file_name + f"_{str(i).zfill(2)}.pkl", 'wb') as f:
                        pickle.dump(res, f)
            
            if verbose:
                print(f"complete games: {len(complete_games)}")
            
        else:
            print(f"Failed case id: {config_id}")
            count_failed += 1
    
    print(f"Total count: {count}, Failed count: {count_failed}")
    print(f"Final config count in directory for {num_obj} objects: {count_existing_configs(output_dir, num_obj)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that processes action_mode and max_num_objects."
    )
    
    parser.add_argument(
        '--action_mode',
        type=str,
        required=True,
        help="Specifies the mode of action to perform."
    )
    
    parser.add_argument(
        '--max_num_objects',
        type=int,
        default=4,
        help="Specifies the maximum number of objects allowed (default: 4)."
    )
    
    parser.add_argument(
        '--num_games',
        type=int,
        default=5,
        help="Specifies the number of games to generate (default: 5)."
    )
    
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2500,
        help="Specifies the maximum number of steps to find solutions (default: 2500)."
    )
    
    parser.add_argument(
        '--verbose',
        default=False,
        action='store_true',
        help="Specifies whether to print verbose output (default: False)."
    )
    
    parser.add_argument(
        '--max-sol-num',
        type=int,
        default=5,
        help="Specifies the maximum number of solutions to generate (default: 5)."        
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help="Specifies the random seed (default: 0)."
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help="Specifies the output directory (default: ./data)."
    )
    
    args = parser.parse_args()
    main(args)