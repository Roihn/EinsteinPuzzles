from environment import *
import pickle
import re

PLAYER_REACH = [
    ['bottom_left_bin', 'player1_bin', 'bottom_right_bin', 'commonbin'],
    ['top_left_bin', 'player2_bin', 'top_right_bin', 'commonbin']
]

BIN_TO_BINID = {
    Bins.PLAYER_1_BIN: 'player1_bin',
    Bins.PLAYER_2_BIN: 'player2_bin',
    Bins.COMMON_BIN: 'commonbin',
    Bins.TOP_LEFT_BIN: 'top_left_bin',
    Bins.TOP_RIGHT_BIN: 'top_right_bin',
    Bins.BOTTOM_LEFT_BIN: 'bottom_left_bin',
    Bins.BOTTOM_RIGHT_BIN: 'bottom_right_bin'
}

RELATION_TO_ID = {
    RelationType.ROW: 'row',
    RelationType.COLUMN: 'column',
    RelationType.DIAGONAL: 'diagonal',
    RelationType.BIN: 'bin'
}


def decode_game_info(path):
    """
    Decode the game information from the pickle file
    the game information follows the format:
    game_info = {
                "states": List[State()],
                "final_locations": Dict[Block(), Bins()],
                "knowledge1": Knowledge(),
                "knowledge2": Knowledge(),
                "moves": List[Move() | Relation() | None],
                "first_player": Perspective(),
            }
    """
    with open(path, 'rb') as f:
        game_info = pickle.load(f)
    # get the game start state
    states = game_info["states"]
    # get the final locations
    final_locations = game_info["final_locations"]
    # get the knowledge of the two players
    knowledge1 = game_info["knowledge1"]
    knowledge2 = game_info["knowledge2"]
    # get the moves
    moves = game_info["moves"]
    
    return states, final_locations, knowledge1, knowledge2, moves

def move_to_str(move):
    if isinstance(move, Move):
        return f"move block{int(move.block.value) + 1} from {BIN_TO_BINID[move.current_bin]} to {BIN_TO_BINID[move.destination_bin]}"
    if isinstance(move, Relation) or isinstance(move, Grounding):
        return f"share {knowledge_to_str(move)}"
    if isinstance(move, Block):
        return f"ask block{int(move.value) + 1}"
    return "pass the turn"

    
def get_first_player(moves):
    step = 0
    moves = moves.copy()
    while len(moves):
        move = moves.pop(0)
        p1_count, p2_count = 0, 0
        if isinstance(move, Move):
            bin0 = move.current_bin
            bin1 = move.destination_bin
            if bin0 in [Bins.PLAYER_1_BIN, Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN, Bins.COMMON_BIN]:
                p1_count += 1
            if bin1 in [Bins.PLAYER_1_BIN, Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN, Bins.COMMON_BIN]:
                p1_count += 1
            if bin0 in [Bins.PLAYER_2_BIN, Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN, Bins.COMMON_BIN]:
                p2_count += 1
            if bin1 in [Bins.PLAYER_2_BIN, Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN, Bins.COMMON_BIN]:
                p2_count += 1

            if p1_count > p2_count:
                # $step is player 1
                if step % 2 == 0:
                    return Perspective.PLAYER_1
                else:
                    return Perspective.PLAYER_2
            elif p1_count < p2_count:
                # $step is player 2
                if step % 2 == 0:
                    return Perspective.PLAYER_2
                else:
                    return Perspective.PLAYER_1
        step += 1

def process_step_info(states, init_knowledge1, init_knowledge2, moves, first_player):
    steps = []
    cur_player = first_player
    knowledge1 = init_knowledge1.copy()
    knowledge2 = init_knowledge2.copy()
    for i, state in enumerate(states):
        step_info = {"step": i}
        step_info["state"] = state
        move = moves[i] if i < len(moves) else None
        step_info["knowledge1"] = knowledge1.copy()
        step_info["knowledge2"] = knowledge2.copy()
        step_info["move"] = move
        if cur_player == Perspective.PLAYER_1:
            step_info["player"] = 1
            cur_player = Perspective.PLAYER_2
        else:
            step_info["player"] = 2
            cur_player = Perspective.PLAYER_1
        steps.append(step_info)
        # update the knowledge based on the current move for the next step
        if isinstance(move, Relation):
            if move not in knowledge1.relations:
                knowledge1.relations.append(move)
            if move not in knowledge2.relations:
                knowledge2.relations.append(move)    
        elif isinstance(move, Grounding):
            if move not in knowledge1.grounding:
                knowledge1.grounding = move
            if move not in knowledge2.grounding:
                knowledge2.grounding = move
    
    return steps

def knowledge_to_str(knowledge):
    knowledge_str = ""
    if isinstance(knowledge, Grounding):
        return f"(block{int(knowledge.block.value) + 1}, in, {BIN_TO_BINID[knowledge.bin]})"
    if isinstance(knowledge, Relation):
        return f"(block{int(knowledge.blocks[0].value) + 1}, block{int(knowledge.blocks[1].value) + 1}, same, {RELATION_TO_ID[knowledge.type]})"    
        
    for g in knowledge.groundings:
        # knowledge_str += f"block{int(g.block.value) + 1} should be in {BIN_TO_BINID[g.bin]}\n" # block# should be in bin#
        knowledge_str += f"(block{int(g.block.value) + 1}, in, {BIN_TO_BINID[g.bin]})\n" # block# should be in bin#
    for r in knowledge.relations:
        # knowledge_str += f"block{int(r.blocks[0].value) + 1} is on the same {RELATION_TO_ID[r.type]} as block{int(r.blocks[1].value) + 1}\n" # block# is on the same row/column/diagonal as block#
        knowledge_str += f"(block{int(r.blocks[0].value) + 1}, block{int(r.blocks[1].value) + 1}, same, {RELATION_TO_ID[r.type]})\n" # block# is on the same row/column/diagonal as block#
    return knowledge_str

def move_history_perspective(move_history, perspective):
    move_history_str = []
    if len(move_history) == 0:
        return "This is the first turn."
    for player_perspective, move, flag in move_history:
        if perspective == player_perspective:
            move_history_str.append(f"You: {move_to_str(move)}")
        else:
            move_history_str.append(f"Partner: {move_to_str(move)}")
        
        if flag is False:
            move_history_str[-1] += ", but failed."
    return "\n".join(move_history_str)


def extract_think_action(input_string, use_cot=True):
    """
    Extract reasoning and action from a string with format:
    <THINK><your reasoning></THINK><ACTION><your action></ACTION>
    
    Args:
        input_string (str): The input string to parse
        use_cot (bool): Flag indicating whether to extract reasoning (Chain of Thought)
        
    Returns:
        dict: A dictionary containing 'reasoning' and 'action' keys
    """
    # Initialize result dictionary
    result = {
        "reasoning": "",
        "action": ""
    }
    
    # Extract action (always needed)
    action_regex = r'<ACTION>([\s\S]*?)</ACTION>'
    action_match = re.search(action_regex, input_string)
    
    if action_match:
        result["action"] = action_match.group(1).strip()
    
    # Extract reasoning only if use_cot is True
    if use_cot:
        think_regex = r'<THINK>([\s\S]*?)</THINK>'
        think_match = re.search(think_regex, input_string)
        
        if think_match:
            result["reasoning"] = think_match.group(1).strip()
    
    return result

def str_to_move(move_str, use_cot = False):
    """
    Convert a string command into a Move, Knowledge, or request object.
    
    The string format should be: <ACTION>action</ACTION>
    
    Supported actions:
    - Move block: "move <block> from <bin> to <bin>"
    - Share knowledge: "share <knowledge>"
    - Request knowledge: "ask <block>"
    - Pass your turn: "pass"
    
    where we have the following bin names:
    - player1_bin
    - player2_bin
    - commonbin
    - top_left_bin
    - top_right_bin
    - bottom_left_bin
    - bottom_right_bin
    
    and the following knowledge types:
    - (<block1>, <block2>, same, row)
    - (<block1>, <block2>, same, column)
    - (<block1>, <block2>, same, diagonal)
    - (<block1>, <block2>, same, bin)
    - (<block1>, in, <bin>)
    
    and the following block names:
    - block0
    - block1
    - block2
    ...
    """
    try:
        res = extract_think_action(move_str, use_cot)
        action_str = res["action"].strip()
    except:
        raise ValueError(f"Invalid format. Expected {'<THINK><reasoning></THINK>' if use_cot else ''}<ACTION>action</ACTION>")
    
    # Pass turn
    if action_str == "pass":
        return "pass"
    
    # Request knowledge
    if action_str.startswith("ask "):
        block_name = action_str[4:].strip()
        # Convert block name to Block enum
        if block_name.startswith("block"):
            try:
                block_num = int(block_name[5:]) - 1
                block = Block(block_num)
                return block
            except ValueError:
                raise ValueError(f"Invalid block name: {block_name}")
        else:
            raise ValueError(f"Invalid block name format: {block_name}")
    
    # Share knowledge
    if action_str.startswith("share "):
        knowledge_str = action_str[6:].strip()
        
        # Parse knowledge string format "(<block1>, <block2>, same, row)" or "(<block1>, in, <bin>)"
        if knowledge_str.startswith("(") and knowledge_str.endswith(")"):
            # Remove parentheses and split by commas
            parts = [p.strip() for p in knowledge_str[1:-1].split(",")]
            
            # Case: (<block1>, in, <bin>)
            if len(parts) == 3 and parts[1] == "in":
                block_name = parts[0]
                bin_name = parts[2]
                
                # Convert block name to Block enum
                if block_name.startswith("block"):
                    try:
                        block_num = int(block_name[5:]) - 1
                        block = Block(block_num)
                    except ValueError:
                        raise ValueError(f"Invalid block name: {block_name}")
                else:
                    raise ValueError(f"Invalid block name format: {block_name}")
                
                # Convert bin name to Bins enum
                bin_map = {
                    "player1_bin": Bins.PLAYER_1_BIN,
                    "player2_bin": Bins.PLAYER_2_BIN,
                    "commonbin": Bins.COMMON_BIN,
                    "top_left_bin": Bins.TOP_LEFT_BIN,
                    "top_right_bin": Bins.TOP_RIGHT_BIN,
                    "bottom_left_bin": Bins.BOTTOM_LEFT_BIN,
                    "bottom_right_bin": Bins.BOTTOM_RIGHT_BIN
                }
                
                if bin_name in bin_map:
                    bin_enum = bin_map[bin_name]
                    return Grounding(block, bin_enum)
                else:
                    raise ValueError(f"Invalid bin name: {bin_name}")
            
            # Case: (<block1>, <block2>, same, row/column/diagonal)
            elif len(parts) == 4 and parts[2] == "same":
                block1_name = parts[0]
                block2_name = parts[1]
                relation_type = parts[3]
                
                # Convert block names to Block enums
                if block1_name.startswith("block") and block2_name.startswith("block"):
                    try:
                        block1_num = int(block1_name[5:]) - 1
                        block2_num = int(block2_name[5:]) - 1
                        block1 = Block(block1_num)
                        block2 = Block(block2_num)
                    except ValueError:
                        raise ValueError(f"Invalid block names: {block1_name}, {block2_name}")
                else:
                    raise ValueError(f"Invalid block name format: {block1_name} or {block2_name}")
                
                # Convert relation type to RelationType enum
                relation_map = {
                    "row": RelationType.ROW,
                    "column": RelationType.COLUMN,
                    "diagonal": RelationType.DIAGONAL,
                    "bin": RelationType.BIN
                }
                
                if relation_type in relation_map:
                    relation_type_enum = relation_map[relation_type]
                    return Relation(block1, block2, relation_type_enum, RelationValue.SAME)
                else:
                    raise ValueError(f"Invalid relation type: {relation_type}")
            else:
                raise ValueError(f"Invalid knowledge format: {knowledge_str}")
        else:
            raise ValueError(f"Invalid knowledge format: {knowledge_str}")
    
    # Move block
    if action_str.startswith("move "):
        # Format: "move <block> from <bin> to <bin>"
        parts = action_str.split()
        if len(parts) == 6 and parts[0] == "move" and parts[2] == "from" and parts[4] == "to":
            block_name = parts[1]
            current_bin_name = parts[3]
            destination_bin_name = parts[5]
            
            # Convert block name to Block enum
            if block_name.startswith("block"):
                try:
                    block_num = int(block_name[5:]) - 1
                    block = Block(block_num)
                except ValueError:
                    raise ValueError(f"Invalid block name: {block_name}")
            else:
                raise ValueError(f"Invalid block name format: {block_name}")
            
            # Convert bin names to Bins enum
            bin_map = {
                "player1_bin": Bins.PLAYER_1_BIN,
                "player2_bin": Bins.PLAYER_2_BIN,
                "commonbin": Bins.COMMON_BIN,
                "top_left_bin": Bins.TOP_LEFT_BIN,
                "top_right_bin": Bins.TOP_RIGHT_BIN,
                "bottom_left_bin": Bins.BOTTOM_LEFT_BIN,
                "bottom_right_bin": Bins.BOTTOM_RIGHT_BIN
            }
            
            if current_bin_name in bin_map and destination_bin_name in bin_map:
                current_bin = bin_map[current_bin_name]
                destination_bin = bin_map[destination_bin_name]
                return Move(block, current_bin, destination_bin)
            else:
                raise ValueError(f"Invalid bin names: {current_bin_name}, {destination_bin_name}")
        else:
            raise ValueError(f"Invalid move format: {action_str}")
    
    # If none of the patterns match
    raise ValueError(f"Invalid command format: {action_str}")


def can_complete_game(all_given_knowledge: Knowledge, final_locations: dict, num_blocks: int) -> bool:
   """
   Check if the game can be completed with the all of the given knowledge.


   Args:
       all_given_knowledge: all of the knowledge before partition between players
       final_locations: {block: bin}
       num_blocks: number of blocks in the game
   """

   unified_player = Player(perspective=Perspective.PLAYER_1, knowledge=all_given_knowledge, num_blocks=num_blocks)
  
   for block, bin in final_locations.items():
       if unified_player.infered_knowledge[block][bin].value == RelationValue.NONE:
           return False
  
   return True

def parse_filename(filename: str):
    """
    Reverse engineers a filename to extract the original configuration parameters.
    
    Args:
        filename: The filename string (e.g., "4_0022_0123_0001_102_00.pkl")
        
    Returns:
        Tuple containing:
        - num_obj: Integer representing number of objects
        - starting_bins_choice: List of Bins enums
        - final_locations_choice: List of Bins enums
        - rel_ids: List of integers
        - knowledge_disparity: List of integers
    """
    # remove prefix path if present
    filename = filename.split('/')[-1]
    
    # Remove file extension if present
    if '.' in filename:
        filename = filename.split('.')[0]
    
    # Split by underscores to get components
    parts = filename.split('_')
    
    # Extract num_obj (first part)
    num_obj = int(parts[0])
    
    # Extract starting_bins_choice (convert integers to Bins enums)
    starting_bins_str = parts[1]
    starting_bins_choice = [Bins(int(x)) for x in starting_bins_str]
    
    # Extract final_locations_choice (convert integers to tuple-based Bins enums)
    final_locations_str = parts[2]
    final_locations_choice = []
    for i in range(len(final_locations_str)):
        value = int(final_locations_str[i])
        # Convert back to the destination bin
        # The mapping was: x.value[0].value + 2*x.value[1].value
        # So we need to determine the row and column from the value
        row_value = value % 2
        column_value = (value - row_value) // 2
        
        row = Rows.TOP if row_value == 0 else Rows.BOTTOM
        column = Columns.LEFT if column_value == 0 else Columns.RIGHT
        
        # Find the corresponding Bins enum
        if row == Rows.TOP and column == Columns.LEFT:
            bin_enum = Bins.TOP_LEFT_BIN
        elif row == Rows.TOP and column == Columns.RIGHT:
            bin_enum = Bins.TOP_RIGHT_BIN
        elif row == Rows.BOTTOM and column == Columns.LEFT:
            bin_enum = Bins.BOTTOM_LEFT_BIN
        else:  # row == Rows.BOTTOM and column == Columns.RIGHT
            bin_enum = Bins.BOTTOM_RIGHT_BIN
            
        final_locations_choice.append(bin_enum)
    
    # Extract rel_ids
    rel_ids_str = parts[3]
    rel_ids = [int(x) for x in rel_ids_str]
    
    # Extract knowledge_disparity
    knowledge_disparity = []
    if len(parts) > 4:
        knowledge_disp_str = parts[4]
        knowledge_disparity = [int(x) for x in knowledge_disp_str]
    
    return (num_obj, starting_bins_choice, final_locations_choice, rel_ids, knowledge_disparity)


def parse_and_reconstruct_config(filename: str) -> tuple:
    """
    Parses a filename and reconstructs the full configuration tuple with Block objects.
    
    Args:
        filename: The filename string (e.g., "4_0022_0123_0001_102_00.pkl")
        
    Returns:
        Tuple containing the complete configuration (num_obj, starting_bins_choice, 
        final_locations_choice, rel_ids, knowledge_disparity)
    """
    num_obj, starting_bins_choice, final_locations_choice, rel_ids, knowledge_disparity = parse_filename(filename)
    
    # Create a complete configuration tuple
    config = (
        num_obj,
        starting_bins_choice,
        final_locations_choice,
        rel_ids,
        knowledge_disparity
    )
    
    return config