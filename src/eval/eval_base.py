import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import permutations
from random import seed, shuffle, choice
from copy import deepcopy
from utils import *
from arguments import parser
from environment import Block, Bins, Grounding, Move, Knowledge, Perspective, Relation, State, Player
from verifier.verifier import reasoning_verifier, communication_verifier, affordance_verifier, calc_error

def partial_success_rate(gt_final_state : State, final_state : State):
    """
    Calculate the partial success rate of the final state
    :param gt_final_state: ground truth final state
    :param final_state: final state
    :return: partial success rate
    """
    gt_bins = gt_final_state.bins
    bins = final_state.bins
    num_blocks = 0
    num_correct = 0
    for bin in [
                Bins.TOP_LEFT_BIN,
                Bins.TOP_RIGHT_BIN,
                Bins.BOTTOM_LEFT_BIN,
                Bins.BOTTOM_RIGHT_BIN,
            ]:
        num_blocks += len(gt_bins[bin])
        num_correct += len([block for block in gt_bins[bin] if block in bins[bin]])
    
    return num_correct, num_blocks


class EinsteinGame():
    def __init__(self, 
                 init_state : State,
                 knowledge1: Knowledge,
                 knowledge2: Knowledge,
                 final_state : State, 
                 is_whole_dialog : bool,
                 action_mode : str,
                 num_blocks : int,
                 use_cot: bool = False,
                 allow_random_guess: bool = False,
                 step_limit : int = 30):
        self.cur_state = init_state
        self.knowledge1 = knowledge1
        self.knowledge2 = knowledge2
        self.num_blocks = num_blocks
        self.final_state = final_state
        self.is_whole_dialog = is_whole_dialog
        self.action_mode = action_mode
        self.step_limit = step_limit
        self.use_cot = use_cot
        self.allow_random_guess = allow_random_guess
        
        self.step_count = 0
        self.move_history = []
        self.players = [Player(Perspective.PLAYER_1, knowledge1, num_blocks=num_blocks), Player(Perspective.PLAYER_2, knowledge2, num_blocks=num_blocks)]
        self.cur_player = self.players[0]
        
    
    def step(self, action_str):
        """
        Execute the requested action and return the new state, reward, terminated, truncated, info
        """
        try:
            action = str_to_move(move_str=action_str, use_cot=self.use_cot)
        except Exception as e:
            print("#"*60, "error:", e, "#"*60)
            print("#"*60, "action_str:", action_str, "#"*60)
            action = None
        print("#"*60, "action:", action, "#"*60)
        new_state, reward, terminated, truncated, info = self.execute(action)
        self.cur_state = new_state
        # Check final state
        if new_state.is_final(self.final_state):
            terminated = True
            reward = 1
        
        self.step_count += 1
        
        if self.step_count >= self.step_limit:
            truncated = True
            
        self.move_history += [(self.cur_player.perspective, action, reward >= 0)]
        if self.is_whole_dialog:
            self.update_assistant(action_str)
        self.cur_player = self.players[1] if self.cur_player.perspective == Perspective.PLAYER_1 else self.players[0]
        
        return new_state, reward, terminated, truncated, info
    
    def execute(self, action):
        reward, terminated, truncated, info = 0, False, False, {}
        player = self.cur_player
        partner = self.players[1-player.perspective.value]
        moves, share_dlg, ask_dlg = player.get_moves(self.cur_state)
        if self.action_mode == "provide_seek":
            all_moves = moves + share_dlg  + ask_dlg
        elif self.action_mode == "provide":
            all_moves = moves + share_dlg
        elif self.action_mode == "seek":
            all_moves = moves + ask_dlg
            if len(self.move_history) > 0:
                last_move = self.move_history[-1][1]
                if isinstance(last_move, Block):
                    # the partner asks for the block
                    block = last_move
                    for share in share_dlg:
                        if isinstance(share, Relation):
                            if share.blocks[0] == block or share.blocks[1] == block:
                                all_moves.append(share)
                        elif isinstance(share, Grounding):
                            if share.block == block:
                                all_moves.append(share)
        else:
            all_moves = moves
        # print("all_moves:", all_moves)
        
        if action not in all_moves:
            # invalid move or redundant/invalid knowledge sharing
            print("action not in all_moves")
            reward = -0.1
            new_state = deepcopy(self.cur_state)
            if isinstance(action, Move) and self.allow_random_guess:
                # check if the action is valid or not (e.g. moving a block from one bin to another)
                ## Check the source and destination
                block, src, dst = action.block, action.current_bin, action.destination_bin
                flag = block in self.cur_state.bins[src] # check if the block is in the source bin
                flag = flag and (src in player.reachable_bins) # check if the source bin is reachable
                flag = flag and (dst in player.reachable_bins) # check if the destination bin is reachable
                
                if flag: # this is a move that is allowed in the environment
                    cur_state = deepcopy(self.cur_state)
                    new_state = player.make_move(cur_state, action, self.final_state, partner)
                    # print('new_state:', new_state)
                    if not cur_state == cur_state:
                        reward = 0 # not label this as a wrong action since it put the block in the correct place by luck
                    # print("reward:", reward)
            return new_state, reward, terminated, truncated, info
        
        if isinstance(action, Move):
            new_state = player.make_move(self.cur_state, action, self.final_state, partner)
            return new_state, reward, terminated, truncated, info
        
        if isinstance(action, Block):
            new_state = deepcopy(self.cur_state)
            return new_state, reward, terminated, truncated, info
        
        if isinstance(action, Relation) or isinstance(action, Grounding):
            new_state = deepcopy(self.cur_state)
            partner = self.players[1] if player.perspective == Perspective.PLAYER_1 else self.players[0]
            player.share_knowledge(action, partner)
            
            return new_state, reward, terminated, truncated, info
        
        raise ValueError(f"Invalid action: {action}")

    def update_assistant(self, action_str):
        assert self.is_whole_dialog, "update_assistant should only be called when is_whole_dialog is True"
        self.dialog_history[self.cur_player.perspective.value].append({
            "role": "assistant",
            "content": action_str
        })

    def render(self):
        side = "bottom" if self.cur_player.perspective == Perspective.PLAYER_1 else "top"
        if self.use_cot:
            if self.action_mode == "provide_seek":
                from prompt.selfplay_llama_cot_share_ask import SYSTEM_PROMPT, USER_PROMPT
            elif self.action_mode == "provide":
                from prompt.selfplay_llama_cot_share import SYSTEM_PROMPT, USER_PROMPT
            elif self.action_mode == "seek":
                from prompt.selfplay_llama_cot_ask import SYSTEM_PROMPT, USER_PROMPT
            elif self.action_mode == "none":
                from prompt.selfplay_llama_cot_none import SYSTEM_PROMPT, USER_PROMPT
        else:
            if self.action_mode == "provide_seek":
                from prompt.selfplay_llama_provide_seek import SYSTEM_PROMPT, USER_PROMPT
            elif self.action_mode == "provide":
                from prompt.selfplay_llama_share import SYSTEM_PROMPT, USER_PROMPT
            elif self.action_mode == "seek":
                from prompt.selfplay_llama_ask import SYSTEM_PROMPT, USER_PROMPT
                # raise ValueError("ask mode is not supported in ask mode")
            elif self.action_mode == "none":
                from prompt.selfplay_llama_none import SYSTEM_PROMPT, USER_PROMPT
                
        if self.is_whole_dialog:
            dialog_history = self.dialog_history[self.cur_player.perspective.value]
        knowledge = self.cur_player.knowledge
        knowledge_str = knowledge_to_str(knowledge)
        blocks = ""
        state = self.cur_state
        for bin in [
                    Bins.TOP_LEFT_BIN,
                    Bins.PLAYER_2_BIN,
                    Bins.TOP_RIGHT_BIN,
                    Bins.COMMON_BIN,
                    Bins.BOTTOM_LEFT_BIN,
                    Bins.PLAYER_1_BIN,                    
                    Bins.BOTTOM_RIGHT_BIN,
                ]:
            blocks += f"{BIN_TO_BINID[bin]}: [{', '.join([f'block{int(b.value) + 1}' for b in state.bins[bin]])}]\n"
        
        reachable_bins = PLAYER_REACH[self.cur_player.perspective.value]
        
        move_history_str = move_history_perspective(self.move_history, self.cur_player.perspective)
        if self.is_whole_dialog:
            dialog_history.append({
            "role": "user",
            "content": USER_PROMPT.format(player_id=self.cur_player.perspective.value + 1, 
                                        side=side, 
                                        knowledge=knowledge_str, 
                                        blocks=blocks, 
                                        bins=reachable_bins, 
                                        move_history=move_history_str)
        })
        else:
            dialog_history = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": USER_PROMPT.format(player_id=self.cur_player.perspective.value + 1, 
                                                side=side, 
                                                knowledge=knowledge_str, 
                                                blocks=blocks, 
                                                bins=reachable_bins, 
                                                move_history=move_history_str)
                }
            ]
        return dialog_history


class EinsteinGameWithVerifier(EinsteinGame):
    def __init__(self, 
                 init_state : State,
                 knowledge1: Knowledge,
                 knowledge2: Knowledge,
                 final_state : State, 
                 is_whole_dialog : bool,
                 action_mode : str,
                 num_blocks : int,
                 use_cot: bool = False,
                 allow_random_guess: bool = False,
                 step_limit : int = 30,
                 verifier: str = "reasoning_verifier"):
        super().__init__(init_state, knowledge1, knowledge2, final_state, is_whole_dialog, action_mode, num_blocks, use_cot, allow_random_guess, step_limit)
        if verifier == "reasoning_verifier":
            self.verifier = reasoning_verifier
        elif verifier == "communication_verifier":
            self.verifier = communication_verifier
        elif verifier == "affordance_verifier":
            self.verifier = affordance_verifier
    
    def step(self, action_str_list):
        """
        Execute the requested action and return the new state, reward, terminated, truncated, info
        """
        final_action = None
        actions_list = []
        actions_err = []
        for action_str in action_str_list:
            try:
                action = str_to_move(move_str=action_str, use_cot=self.use_cot)
            except Exception as e:
                print("#"*60, "error:", e, "#"*60)
                print("#"*60, "action_str:", action_str, "#"*60)
                action = None
            actions_list.append(action)
        
        # Rate the sampled actions and related errors
        for action in actions_list:
            is_valid = self.verifier(self.cur_player, 
                                        self.cur_state, 
                                        self.action_mode, 
                                        action)
            actions_err.append(calc_error(action) if not is_valid else 0)
        
        # Pick the action with the least error
        sum_err = 0
        used_action_idx = -1
        corrected = False
        for i, (action, err) in enumerate(zip(actions_list, actions_err)):
            if err == 0:
                # Valid action
                final_action = action
                used_action_idx = i
                # Get the sum of errors before it reaches this action
                sum_err = sum(actions_err[:i+1])
                corrected = i != 0 # initially wrong, but corrected by the verifier
                break
        
        for action, err in zip(actions_list, actions_err):
            print("#"*60, "action:", action, "error:", err, "#"*60)

        if final_action is None:
            # pick the action with the least error
            final_action = actions_list[actions_err.index(min(actions_err))]
            sum_err = sum(actions_err)

        action = final_action
        new_state, reward, terminated, truncated, info = self.execute(action)
        self.cur_state = new_state
        # Check final state
        if new_state.is_final(self.final_state):
            terminated = True
            reward = 1
        
        self.step_count += 1
        
        info["used_trials"] = used_action_idx if used_action_idx >= 0 else len(actions_list) - 1
        info["sum_err"] = sum_err
        info["actions_err"] = actions_err
        info["corrected"] = corrected
        
        if self.step_count >= self.step_limit:
            truncated = True
            
        self.move_history += [(self.cur_player.perspective, action, reward >= 0)]
        if self.is_whole_dialog:
            self.update_assistant(action_str)
        self.cur_player = self.players[1] if self.cur_player.perspective == Perspective.PLAYER_1 else self.players[0]
        
        # Take the sum of errors as the reward
        reward -= sum_err
        return new_state, reward, terminated, truncated, info
    
    

class EinsteinGameWithVerifierModelModel(EinsteinGame):
    def __init__(self, 
                 init_state : State,
                 knowledge1: Knowledge,
                 knowledge2: Knowledge,
                 final_state : State, 
                 is_whole_dialog : bool,
                 action_mode1 : str,
                 action_mode2 : str,
                 num_blocks : int,
                 use_cot1: bool = False,
                 use_cot2: bool = False,
                 allow_random_guess: bool = False,
                 step_limit : int = 30,
                 verifier: str = "reasoning_verifier"):
        self.cur_state = init_state
        self.knowledge1 = knowledge1
        self.knowledge2 = knowledge2
        self.num_blocks = num_blocks
        self.final_state = final_state
        self.is_whole_dialog = is_whole_dialog
        self.action_mode1 = action_mode1
        self.action_mode2 = action_mode2
        self.step_limit = step_limit
        self.use_cot1 = use_cot1
        self.use_cot2 = use_cot2
        self.allow_random_guess = allow_random_guess
        
        self.step_count = 0
        self.move_history = []
        self.players = [Player(Perspective.PLAYER_1, knowledge1, num_blocks=num_blocks), Player(Perspective.PLAYER_2, knowledge2, num_blocks=num_blocks)]
        self.cur_player = self.players[0]
        if verifier == "reasoning_verifier":
            self.verifier = reasoning_verifier
        elif verifier == "communication_verifier":
            self.verifier = communication_verifier
        elif verifier == "affordance_verifier":
            self.verifier = affordance_verifier
        
    
    def step(self, action_str_list):
        """
        Execute the requested action and return the new state, reward, terminated, truncated, info
        """
        final_action = None
        actions_list = []
        actions_err = []
        for action_str in action_str_list:
            try:
                action = str_to_move(move_str=action_str, use_cot=self.use_cot)
            except Exception as e:
                print("#"*60, "error:", e, "#"*60)
                print("#"*60, "action_str:", action_str, "#"*60)
                action = None
            actions_list.append(action)
        
        # Rate the sampled actions and related errors
        for action in actions_list:
            is_valid = self.verifier(self.cur_player, 
                                        self.cur_state, 
                                        self.action_mode, 
                                        action)
            actions_err.append(calc_error(action) if not is_valid else 0)
        
        # Pick the action with the least error
        sum_err = 0
        used_action_idx = -1
        corrected = False
        for i, (action, err) in enumerate(zip(actions_list, actions_err)):
            if err == 0:
                # Valid action
                final_action = action
                used_action_idx = i
                # Get the sum of errors before it reaches this action
                sum_err = sum(actions_err[:i+1])
                corrected = i != 0 # initially wrong, but corrected by the verifier
                break
        
        for action, err in zip(actions_list, actions_err):
            print("#"*60, "action:", action, "error:", err, "#"*60)

        if final_action is None:
            # pick the action with the least error
            final_action = actions_list[actions_err.index(min(actions_err))]
            sum_err = sum(actions_err)

        action = final_action
        new_state, reward, terminated, truncated, info = self.execute(action)
        self.cur_state = new_state
        # Check final state
        if new_state.is_final(self.final_state):
            terminated = True
            reward = 1
        
        self.step_count += 1
        
        info["used_trials"] = used_action_idx if used_action_idx >= 0 else len(actions_list) - 1
        info["sum_err"] = sum_err
        info["actions_err"] = actions_err
        info["corrected"] = corrected
        
        if self.step_count >= self.step_limit:
            truncated = True
            
        self.move_history += [(self.cur_player.perspective, action, reward >= 0)]
        if self.is_whole_dialog:
            self.update_assistant(action_str)
        self.cur_player = self.players[1] if self.cur_player.perspective == Perspective.PLAYER_1 else self.players[0]
        
        # Take the sum of errors as the reward
        reward -= sum_err
        return new_state, reward, terminated, truncated, info
    
    def execute(self, action):
        reward, terminated, truncated, info = 0, False, False, {}
        player = self.cur_player
        partner = self.players[1-player.perspective.value]
        moves, share_dlg, ask_dlg = player.get_moves(self.cur_state)
        if self.action_mode == "provide_seek":
            all_moves = moves + share_dlg  + ask_dlg
        elif self.action_mode == "provide":
            all_moves = moves + share_dlg
        elif self.action_mode == "seek":
            all_moves = moves + ask_dlg
        else:
            all_moves = moves
        # print("all_moves:", all_moves)
        
        if action not in all_moves:
            # invalid move or redundant/invalid knowledge sharing
            print("action not in all_moves")
            reward = -0.1
            new_state = deepcopy(self.cur_state)
            if isinstance(action, Move) and self.allow_random_guess:
                # check if the action is valid or not (e.g. moving a block from one bin to another)
                ## Check the source and destination
                block, src, dst = action.block, action.current_bin, action.destination_bin
                flag = block in self.cur_state.bins[src] # check if the block is in the source bin
                flag = flag and (src in player.reachable_bins) # check if the source bin is reachable
                flag = flag and (dst in player.reachable_bins) # check if the destination bin is reachable
                
                if flag: # this is a move that is allowed in the environment
                    cur_state = deepcopy(self.cur_state)
                    new_state = player.make_move(cur_state, action, self.final_state, partner)
                    # print('new_state:', new_state)
                    if not cur_state == cur_state:
                        reward = 0 # not label this as a wrong action since it put the block in the correct place by luck
                    # print("reward:", reward)
            return new_state, reward, terminated, truncated, info
        
        if isinstance(action, Move):
            new_state = player.make_move(self.cur_state, action, self.final_state, partner)
            return new_state, reward, terminated, truncated, info
        
        if isinstance(action, Block):
            new_state = deepcopy(self.cur_state)
            return new_state, reward, terminated, truncated, info
        
        if isinstance(action, Relation) or isinstance(action, Grounding):
            new_state = deepcopy(self.cur_state)
            partner = self.players[1] if player.perspective == Perspective.PLAYER_1 else self.players[0]
            player.share_knowledge(action, partner)
            
            return new_state, reward, terminated, truncated, info
        
        raise ValueError(f"Invalid action: {action}")

    def render(self):
        side = "bottom" if self.cur_player.perspective == Perspective.PLAYER_1 else "top"
        self.action_mode = self.action_mode1 if self.cur_player.perspective == Perspective.PLAYER_1 else self.action_mode2
        self.use_cot = self.use_cot1 if self.cur_player.perspective == Perspective.PLAYER_1 else self.use_cot2
        if self.use_cot:
            if self.action_mode == "provide_seek":
                from prompt.selfplay_llama_cot_share_ask import SYSTEM_PROMPT, USER_PROMPT
            elif self.action_mode == "provide":
                from prompt.selfplay_llama_cot_share import SYSTEM_PROMPT, USER_PROMPT
            elif self.action_mode == "seek":
                from prompt.selfplay_llama_cot_ask import SYSTEM_PROMPT, USER_PROMPT
            elif self.action_mode == "none":
                from prompt.selfplay_llama_cot_none import SYSTEM_PROMPT, USER_PROMPT
        else:
            if self.action_mode == "provide_seek":
                from prompt.selfplay_llama_provide_seek import SYSTEM_PROMPT, USER_PROMPT
            elif self.action_mode == "provide":
                from prompt.selfplay_llama_share import SYSTEM_PROMPT, USER_PROMPT
            elif self.action_mode == "seek":
                from prompt.selfplay_llama_ask import SYSTEM_PROMPT, USER_PROMPT
                # raise ValueError("ask mode is not supported in ask mode")
            elif self.action_mode == "none":
                from prompt.selfplay_llama_none import SYSTEM_PROMPT, USER_PROMPT
                
        if self.is_whole_dialog:
            raise ValueError("is_whole_dialog is not supported in this game")
        knowledge = self.cur_player.knowledge
        knowledge_str = knowledge_to_str(knowledge)
        blocks = ""
        state = self.cur_state
        for bin in [
                    Bins.TOP_LEFT_BIN,
                    Bins.PLAYER_2_BIN,
                    Bins.TOP_RIGHT_BIN,
                    Bins.COMMON_BIN,
                    Bins.BOTTOM_LEFT_BIN,
                    Bins.PLAYER_1_BIN,                    
                    Bins.BOTTOM_RIGHT_BIN,
                ]:
            blocks += f"{BIN_TO_BINID[bin]}: [{', '.join([f'block{int(b.value) + 1}' for b in state.bins[bin]])}]\n"
        
        reachable_bins = PLAYER_REACH[self.cur_player.perspective.value]
        
        move_history_str = move_history_perspective(self.move_history, self.cur_player.perspective)
    
        dialog_history = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(player_id=self.cur_player.perspective.value + 1, 
                                            side=side, 
                                            knowledge=knowledge_str, 
                                            blocks=blocks, 
                                            bins=reachable_bins, 
                                            move_history=move_history_str)
            }
        ]
        return dialog_history
