import os
import sys
from copy import deepcopy

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment import Player, State, Move, Block, Bins, Perspective

def reasoning_verifier(player:Player, state:State, action_mode:str, action):
    """
    Returns the reasoning action for the player given the current state
    """
    if action is None:
        return False

    moves, share_dlg, ask_dlg = player.get_moves(state)
    if action_mode == "provide_seek":
        all_moves = moves + share_dlg + ask_dlg
    elif action_mode == "provide":
        all_moves = moves + share_dlg
    elif action_mode == "seek":
        all_moves = moves + ask_dlg
    else:
        all_moves = moves
    
    if len(all_moves) > 0:
        return action in all_moves
    
    if not (isinstance(action, Move) or action == "pass"):
        return False
    
    if isinstance(action, Move):
        # player may make a random guess; we need to identify if it is a valid move or not
        # check if the source and destination bin of the move is reachable to the player
        reachable_bins = [Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN, Bins.COMMON_BIN, Bins.PLAYER_1_BIN] if player.perspective == Perspective.PLAYER_1 else [Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN, Bins.COMMON_BIN, Bins.PLAYER_2_BIN]
        flag = True
        flag = flag and (action.current_bin in reachable_bins)               # check if the source bin is reachable
        flag = flag and (action.destination_bin in reachable_bins)          # check if the destination bin is reachable
        flag = flag and (action.block in state.bins[action.current_bin])     # check if the block is in the source bin
        flag = flag and (action.current_bin != action.destination_bin)       # check if the source and destination bin are not the same
        return flag
    
    return True

def communication_verifier(player:Player, state:State, action_mode:str, action):
    """
    Return if the communication action is valid
    If the communication action is ask, then as long as it is not asking about an already placed block, it is valid
    If the communication action is share, then it needs to check if the constraint is explicitly stated, and if it has been shared before or not.
    """
    if action == "pass":
        return True
    
    assert action_mode != "none", "action_mode should not be none"
    if isinstance(action, Move):
        # If the action is a move, then no need to verify
        return True

    if isinstance(action, Block):
        # ask
        # check if the block is already placed
        return action not in [Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN, Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN]

    # share
    knowledge_is_known = action in player.knowledge.relations or action in player.knowledge.groundings
    if not knowledge_is_known:
        # sharing an unknown knowledge
        return False
    # check if the knowledge is alread shared
    return not player.shared_knowledge_flags[action]


def calc_error(action):
    """
    Returns the number of steps wasted by the wrong action
    """
    return 1 if not isinstance(action, Block) else 2


def affordance_verifier(player:Player, state:State, action_mode:str, action):
    """
    Returns whether the action obeys the affordance rules
    """
    if not isinstance(action, Move):
        return True

    reachable_bins = [Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN, Bins.COMMON_BIN, Bins.PLAYER_1_BIN] if player.perspective == Perspective.PLAYER_1 else [Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN, Bins.COMMON_BIN, Bins.PLAYER_2_BIN]
    flag = True
    flag = flag and (action.current_bin in reachable_bins)               # check if the source bin is reachable
    flag = flag and (action.destination_bin in reachable_bins)          # check if the destination bin is reachable
    flag = flag and (action.block in state.bins[action.current_bin])     # check if the block is in the source bin
    flag = flag and (action.current_bin != action.destination_bin)       # check if the source and destination bin are not the same
    return flag