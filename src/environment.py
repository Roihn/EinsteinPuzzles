from enum import Enum
from typing import Union
from copy import deepcopy
import json
from collections import defaultdict

def str_hash(x):
    return int.from_bytes(str(x).encode(), 'little')

class Perspective(Enum):
    PLAYER_1 = 0
    PLAYER_2 = 1

class Columns(Enum):
    LEFT    = 0
    RIGHT   = 1

class Rows(Enum):
    TOP     = 0
    BOTTOM  = 1

class Bins(Enum):
    NONE                = -1
    PLAYER_1_BIN        = 0
    COMMON_BIN          = 1
    PLAYER_2_BIN        = 2
    TOP_LEFT_BIN        = (Rows.TOP,    Columns.LEFT)
    TOP_RIGHT_BIN       = (Rows.TOP,    Columns.RIGHT)
    BOTTOM_LEFT_BIN     = (Rows.BOTTOM, Columns.LEFT)
    BOTTOM_RIGHT_BIN    = (Rows.BOTTOM, Columns.RIGHT)

STARTING_BINS       = [Bins.PLAYER_1_BIN, Bins.PLAYER_2_BIN]
DESTINATION_BINS    = [Bins. TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN, Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN]

def bin2num(bn : Bins):
    return (2 + bn.value[0].value+2*bn.value[1].value) if isinstance(bn.value, tuple) else bn.value

class BlockColor(Enum):
    RED     = 0
    GREEN   = 1
    BLUE    = 2
    YELLOW  = 3
    PURPLE  = 4

class BlockShape(Enum):
    NONE        = -1
    SQUARE      = 0
    CIRCLE      = 1
    TRIANGLE    = 2
    PENTAGON    = 3
    HEXAGON     = 4
    
class RelationType(Enum):
    NONE        = -1
    BIN         = 0
    ROW         = 1
    COLUMN      = 2
    DIAGONAL    = 3

class RelationValue(Enum):
    NONE        = -1
    SAME        = 0
    DIFFERENT   = 1


class Block(Enum):
    NONE  = -1
    OBJ00 =  0
    OBJ01 =  1
    OBJ02 =  2
    OBJ03 =  3
    OBJ04 =  4
    OBJ05 =  5
    OBJ06 =  6
    OBJ07 =  7
    OBJ08 =  8
    OBJ09 =  9
    OBJ10 = 10
    OBJ11 = 11
    OBJ12 = 12
    OBJ13 = 13
    OBJ14 = 14
    OBJ15 = 15
    OBJ16 = 16
    OBJ17 = 17
    OBJ18 = 18
    OBJ19 = 19
    OBJ20 = 20
    OBJ21 = 21
    OBJ22 = 22
    OBJ23 = 23
    OBJ24 = 24
    OBJ25 = 25
    OBJ26 = 26
    OBJ27 = 27
    OBJ28 = 28
    OBJ29 = 29

# class Block:
#     def __init__(self, color : BlockColor, shape : BlockShape):
#         self.color = color
#         self.shape = shape

#     def __repr__(self):
#         return f'Block(color: {self.color.name}, shape: {self.shape.name})'

#     def __str__(self):
#         return f'{self.color.name}-{self.shape.name}'
    
#     def same_as(self, target):
#         return (self.color.value == target.color.value) and (self.shape.value == target.shape.value)
    
#     def __eq__(self,target):
#         return self.same_as(target)
    
#     def __ne__(self, target):
#         return not self.same_as(target)
    
#     def __hash__(self):
#         return hash(self.__repr__)
    

class Relation:
    def __init__(self, obj1: Block, obj2 : Block, type : RelationType, value : RelationValue):
        assert isinstance(obj1,Block)
        assert isinstance(obj2,Block)
        assert isinstance(type,RelationType)
        assert isinstance(value,RelationValue)
        # self.blocks = sorted([obj1, obj2], key=lambda o: o.color.value*len(BlockColor)+o.shape.value)
        self.blocks = sorted([obj1, obj2], key=lambda o: o.value)
        self.type = type
        self.value = value
        
    def to_tuple(self):
        return (self.blocks[0].value, self.type.value, self.value.value, self.blocks[1].value)

    def __repr__(self):
        return f'Relation({[self.blocks[0], self.value, self.type, self.blocks[1]]})'

    def __str__(self):
        return f'Relation({self.blocks[0]}, {self.value}, {self.type}, {self.blocks[1]})'
    
    def __eq__(a, b):
        return repr(a) == repr(b)
    
    def __ne__(a,b):
        return not a.__eq__(b)
    
    def __hash__(self):
        return str_hash(repr(self))


class Grounding:
    def __init__(self, block : Block, bin : Bins, value: RelationValue = RelationValue.SAME):
        self.block = block
        self.bin = bin
        self.value = value

    def __repr__(self):
        return f'Grounding({[self.block, self.bin, self.value]})'

    def __str__(self):
        return f'Grounding({self.block}, {self.bin.name}, {self.value})'
    
    def __eq__(a, b) -> bool:
        return repr(a) == repr(b)
    
    def __ne__(a, b) -> bool:
        return not a.__eq__(b)
    
    def __hash__(self):
        # Compute hash from the tuple of attributes
        return hash((self.block, self.bin, self.value))
    

class Knowledge:
    def __init__(self, relations : list[Relation], groundings : list[Grounding]):
        if not isinstance(groundings, list):
            groundings = [groundings]
        assert isinstance(groundings, list)
        assert isinstance(relations, list)
        assert all([isinstance(x, Relation) for x in relations])
        assert all([isinstance(x, Grounding) for x in groundings])

        self.relations = relations
        self.groundings = groundings

    def __repr__(self):
        return f'Knowledge({self.groundings + self.relations})'

    def __str__(self):
        return f'Knowledge({", ".join(map(str, self.groundings))}, {", ".join(map(str,self.relations))})'

    def copy(self):
        return Knowledge(deepcopy(self.relations), deepcopy(self.groundings))

class Move:
    def __init__(self, block : Block, current_bin : Bins, destination_bin : Bins):
        self.block = block
        self.current_bin = current_bin
        self.destination_bin = destination_bin
        
    def make(self, state, final_state):
        # get the final bin of the block
        final_bin = None
        state_dict = state.to_dict()
        for bin in final_state.bins:
            if self.block in final_state.bins[bin]:
                final_bin = bin
                break
        # if the move is not valid, return the original state
        if self.destination_bin not in [Bins.COMMON_BIN, final_bin]:
            return State(state_dict)
        state_dict[self.block] = self.destination_bin
        return State(state_dict)
    
    def to_tuple(self):
        return (self.block.value, bin2num(self.current_bin), bin2num(self.destination_bin))
        
    def __repr__(self):
        return f'Move({self.block}, {self.current_bin}, {self.destination_bin})'
        
    def __str__(self):
        return f'Move {self.block} from {self.current_bin} to {self.destination_bin}'
    
    def __eq__(a, b):
        return repr(a) == repr(b)


class State:
    def __init__(self, block_location : dict[Block, Bins]):
        assert all([isinstance(x,Block) for x in block_location.keys()])
        assert all([isinstance(x,Bins) for x in block_location.values()])

        self.bins = {x : [] for x in Bins}
        for block, location in block_location.items():
            self.bins[location].append(block)
            
        for bin_key in self.bins.keys():
            self.bins[bin_key] = sorted(self.bins[bin_key], key=lambda x: x.value)

    def __repr__(self):
        return f'State({[self.bins]})'

    def __str__(self, indent=None):
        return json.dumps({str(k) : list(map(str, v)) for k, v in self.bins.items()}, indent=indent)
    
    def for_vis(self):
        ret_lst = []
        for k in [
                    Bins.PLAYER_2_BIN,
                    Bins.TOP_LEFT_BIN,
                    Bins.TOP_RIGHT_BIN,
                    Bins.COMMON_BIN,
                    Bins.BOTTOM_LEFT_BIN,
                    Bins.BOTTOM_RIGHT_BIN,
                    Bins.PLAYER_1_BIN,
                  ]:
            ret_lst.append(f'{k:22s} {[str(x) for x in self.bins[k]]}')
        return '\n'.join(ret_lst)

    def get_moves(self, perspective : Perspective, knowledge : Knowledge):
        """
        Based on the knowledge and given perspective, return the possible moves for the player
        Here moves only include the physical moves

        """
        assert isinstance(perspective, Perspective)
        assert isinstance(knowledge, Knowledge)
        move_lst = []
        start_bin = eval(f'Bins.{perspective.name}_BIN')
        
        # print(perspective, start_bin, self.bins[start_bin])
        available_bins = [Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN] if perspective == Perspective.PLAYER_1 else [Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN]
        # for bin in [start_bin, Bins.COMMON_BIN]:
        #     # for the bin of the player / shared bin
        #     # print(bin)
        #     # print(self.bins[bin])
        #     for obj in self.bins[bin]:
        #         # for each object in the bin
        #         for grounding in knowledge.groundings:
        #             print(f"comparing {obj} and {grounding} and {grounding.bin} with {grounding.block}")
        #             if obj == grounding.block:
        #                 # kowledge.grounding.block means obj in XX (somewhere)
        #                 # print(knowledge.grounding)
        #                 dest_bin = grounding.bin if grounding.bin in available_bins else Bins.COMMON_BIN
        #                 # move to the target bin / shared bin if available
        #                 move_lst.append(Move(obj, current_bin = bin, destination_bin = dest_bin))
        #                 print("appended")

                # for rel in knowledge.relations:
                #     if obj in rel.blocks:
                #         anchor = list(set(rel.blocks)-set([obj]))[0]
                #         anchor_bin = self.to_dict()[list(set(rel.blocks)-set([obj]))[0]]
                #         destinations = resolveBinRelation(anchor_bin, rel)
                #         if destinations:
                #             if len(destinations) == 1:
                #                 dest_bin = destinations[0] if destinations[0] in available_bins else Bins.COMMON_BIN
                #                 if not bin == dest_bin:
                #                     move_lst.append(Move(obj, current_bin = bin, destination_bin = dest_bin))
                                
                        # print(rel.blocks, anchor, anchor_bin, resolveBinRelation(anchor_bin, rel))
        for grounding in knowledge.groundings:
            if grounding.block in self.bins[grounding.bin]:
                # it has been placed to the right bin
                continue
            # get the current location of the block
            current_bin = None
            for bin in self.bins:
                if grounding.block in self.bins[bin]:
                    current_bin = bin
                    break
            if current_bin in [start_bin, Bins.COMMON_BIN] + available_bins and grounding.bin in available_bins:
                # the block is in the player's bin or the shared bin and the target bin is available
                move_lst.append(Move(grounding.block, current_bin = current_bin, destination_bin = grounding.bin))
            elif current_bin in [start_bin] + available_bins:
                # the block is in the player's bin but the target bin is not available
                move_lst.append(Move(grounding.block, current_bin = current_bin, destination_bin = Bins.COMMON_BIN))
            else:
                # either the block is in the shared bin or the target bin is not available
                continue

        return move_lst
    
    def is_final(self, final_state):
        # print(set(self.bins[Bins.BOTTOM_LEFT_BIN]), set(final_state.bins[Bins.BOTTOM_LEFT_BIN]),set(self.bins[Bins.BOTTOM_LEFT_BIN]) == set(final_state.bins[Bins.BOTTOM_LEFT_BIN]))
        # print(set(self.bins[Bins.BOTTOM_RIGHT_BIN]), set(final_state.bins[Bins.BOTTOM_RIGHT_BIN]),set(self.bins[Bins.BOTTOM_RIGHT_BIN]) == set(final_state.bins[Bins.BOTTOM_RIGHT_BIN]))
        # print(set(self.bins[Bins.TOP_LEFT_BIN]), set(final_state.bins[Bins.TOP_LEFT_BIN]),set(self.bins[Bins.TOP_LEFT_BIN]) == set(final_state.bins[Bins.TOP_LEFT_BIN]))
        # print(set(self.bins[Bins.TOP_RIGHT_BIN]), set(final_state.bins[Bins.TOP_RIGHT_BIN]),set(self.bins[Bins.TOP_RIGHT_BIN]) == set(final_state.bins[Bins.TOP_RIGHT_BIN]))
        return all([
            set(self.bins[Bins.BOTTOM_LEFT_BIN]) == set(final_state.bins[Bins.BOTTOM_LEFT_BIN]),
            set(self.bins[Bins.BOTTOM_RIGHT_BIN]) == set(final_state.bins[Bins.BOTTOM_RIGHT_BIN]),
            set(self.bins[Bins.TOP_LEFT_BIN]) == set(final_state.bins[Bins.TOP_LEFT_BIN]),
            set(self.bins[Bins.TOP_RIGHT_BIN]) == set(final_state.bins[Bins.TOP_RIGHT_BIN]),
        ])

    def get_cur_progress(self, final_state):
        """
        Get the current progress of the state compared to the final state
        """
        assert isinstance(final_state, State)
        progress = 0
        for bin in [Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN, Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN]:
            progress += len(self.bins[bin])
        return progress

    def __iter__(self):
        yield from sorted([(obj, bin) for bin, obj_list in self.bins.items() for obj in obj_list], key=lambda x: x[0].value)

    def to_list(self):
        return list(self)

    def to_dict(self):
        return dict(self.to_list())
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, State):
            return False
        return all([set(self.bins[bin]) == set(__value.bins[bin]) for bin in self.bins.keys()])


def resolveBinRelation(bin : Bins, relation : Relation):
    assert isinstance(bin, Bins)
    assert isinstance(relation, Relation)

    if bin in [Bins.PLAYER_1_BIN, Bins.COMMON_BIN , Bins.PLAYER_2_BIN]:
        return []
    else:
        row, col = bin.value
        new_col = Columns(1 - col.value)
        new_row = Rows(1 - row.value)
        if relation.type == RelationType.ROW:
            return [Bins((row, new_col))] if relation.value == RelationValue.SAME else [Bins((new_row, new_col)), Bins((new_row, col))]
        elif relation.type == RelationType.COLUMN:
            return [Bins((new_row, col))] if relation.value == RelationValue.SAME else [Bins((new_row, new_col)), Bins((row, new_col))]
        elif relation.type == RelationType.DIAGONAL:
            return [Bins((new_row, new_col))] if relation.value == RelationValue.SAME else [Bins((row, new_col)), Bins((new_row, col))]
        elif relation.type == RelationType.BIN:
            return [Bins((row, col))] if relation.value == RelationValue.SAME else [Bins((new_row, new_col)), Bins((new_row, col)), Bins((row, new_col))]
        else:
            return f'ERROR with | {relation.type} | {relation.value}'

class StateMoveAndKnowledge:
    def __init__(self, state : State, knowledge1 : Knowledge, knowledge2: Knowledge, move):
        assert isinstance(state, State)
        assert isinstance(knowledge1, Knowledge)
        assert isinstance(knowledge2, Knowledge)
        assert isinstance(move, Move) or isinstance(move, Relation) or isinstance(move, Grounding) or move is None
        self.state = state
        self.knowledge1 = knowledge1
        self.knowledge2 = knowledge2
        self.move = move
        
    def __repr__(self):
        return f'StateAndKnowledge({self.state}, {self.knowledge1}, {self.knowledge2}, {self.move})'


class Player:
    def __init__(self, perspective : Perspective, knowledge : Knowledge, num_blocks: int = 4, shared_knowledge_flags : dict = None, infered_knowledge = None, connected_relations_pairs = None):
        assert isinstance(perspective, Perspective)
        assert isinstance(knowledge, Knowledge)
        self.perspective = perspective
        self.knowledge = knowledge
        self.num_blocks = num_blocks
        if shared_knowledge_flags is None:
            self.shared_knowledge_flags = defaultdict(bool)
        else:
            self.shared_knowledge_flags = shared_knowledge_flags

        if self.perspective == Perspective.PLAYER_1:
            self.reachable_bins = [Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN, Bins.COMMON_BIN, Bins.PLAYER_1_BIN]
        else:
            self.reachable_bins = [Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN, Bins.COMMON_BIN, Bins.PLAYER_2_BIN]

        if infered_knowledge is None:
            self.connected_relations_pairs = set()
            self.init_infered_knowledge()
        else:
            assert connected_relations_pairs is not None
            self.connected_relations_pairs = connected_relations_pairs
            self.infered_knowledge = infered_knowledge
        
        # list_infered_knowledge = self.get_infered_knowledge()
        

    def get_moves(self, state : State):
        knowledge = self.get_infered_knowledge()
        # knowledge = self.knowledge
        physical_moves = state.get_moves(self.perspective, knowledge)
        
        knowledge = self.knowledge
        share_moves = [rel for rel in knowledge.relations if not self.shared_knowledge_flags[rel]]
        # print(f"before: {len(share_moves)}")
        share_moves = []
        for relation in knowledge.relations:
            if self.shared_knowledge_flags[relation]:
                continue
            block0, block1 = relation.blocks
            # get the bin of the block0 and block1. If both are in one of the four final bins, then we assume they are already in the right place
            bin0, bin1 = None, None
            for bin in state.bins:
                if block0 in state.bins[bin]:
                    bin0 = bin
                if block1 in state.bins[bin]:
                    bin1 = bin
            if bin0 not in [Bins.PLAYER_1_BIN, Bins.PLAYER_2_BIN, Bins.COMMON_BIN] and bin1 not in [Bins.PLAYER_1_BIN, Bins.PLAYER_2_BIN, Bins.COMMON_BIN]:
                self.shared_knowledge_flags[relation] = True
                continue
            # if the relation is not shared, then we add the relation to the share_moves
            share_moves.append(Relation(block0, block1, relation.type, relation.value))
        # print(f"after: {len(share_moves)}")
        for grouding in knowledge.groundings:
            # if the grounding object is already in the right bin, no need to share the knowledge
            for bin in state.bins:
                if bin == grouding.bin and grouding.block in state.bins[bin]:
                    self.shared_knowledge_flags[grouding] = True
                    break
        share_moves += [grounding for grounding in knowledge.groundings if not self.shared_knowledge_flags[grounding]]
        ask_moves = self.check_unknown_blocks()
        return physical_moves, share_moves, ask_moves

    def check_unknown_blocks(self):
        blocks = [block for block in Block][:self.num_blocks]
        knowledge = self.get_infered_knowledge()
        for grounding in knowledge.groundings:
            # remove all the blocks that are already known
            blocks.remove(grounding.block)
        
        return blocks

    def get_available_moves(self, state : State):
        """
        Only used when there are no moves available according to the knowledge, and have to make a random move.
        If I don't know the final correct location of the block, and I'm able to move it, then I will move it to any available bin.
        """
        unknown_blocks = self.check_unknown_blocks()
        available_source_bins = [Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN, Bins.PLAYER_1_BIN] if self.perspective == Perspective.PLAYER_1 else [Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN, Bins.PLAYER_2_BIN]
        available_source_bins.append(Bins.COMMON_BIN)
        available_target_bins = [Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN, Bins.COMMON_BIN] if self.perspective == Perspective.PLAYER_1 else [Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN, Bins.COMMON_BIN]
        available_moves = []
        for block in unknown_blocks:
            blocks_in_source_bins = [block for bin in available_source_bins for block in state.bins[bin]]
            if block not in blocks_in_source_bins:
                continue
            current_source_bin = None
            for source_bin in available_source_bins:
                if block in state.bins[source_bin]:
                    current_source_bin = source_bin
                    break
            for target_bin in available_target_bins:
                if block not in state.bins[target_bin]:
                    available_moves.append(Move(block, current_source_bin, target_bin))
                    
        return available_moves

    def make_physical_move(self, state : State, move : Move, final_state : State, partner: 'Player' = None):
        new_state = move.make(state, final_state)
        # print("state before move: ", state)
        # print("state after move: ", new_state)
        # print(new_state == state)
        # when a physical move is made, the corresponding knowledge will be updated to both players
        if new_state != state and not move.destination_bin == Bins.COMMON_BIN:
            self.update_knowledge(Grounding(move.block, move.destination_bin))
            if partner:
                partner.update_knowledge(Grounding(move.block, move.destination_bin))
        return new_state
    
    def share_knowledge(self, relation : Union[Relation, Grounding], partner : 'Player'):
        # print("sharing knowledge")
        partner.update_knowledge(relation)
        # the shared knowledge will be explicitly displayed to the partner
        if isinstance(relation, Grounding):
            if not relation in partner.knowledge.groundings:
                partner.knowledge.groundings.append(relation)
        else:
            if not relation in partner.knowledge.relations:
                partner.knowledge.relations.append(relation)
        self.shared_knowledge_flags[relation] = True
        partner.shared_knowledge_flags[relation] = True
            
    def update_knowledge(self, relation):
        if isinstance(relation, Grounding):
            if self.infered_knowledge[relation.block][relation.bin].value == RelationValue.NONE:
                # did not infer this relation before
                self.infered_knowledge[relation.block][relation.bin] = relation
                self.infered_knowledge[relation.bin][relation.block] = relation
                self.infer()
        else:
            if self.infered_knowledge[relation.blocks[0]][relation.blocks[1]].value == RelationValue.NONE:
                # did not infer this relation before
                self.infered_knowledge[relation.blocks[0]][relation.blocks[1]] = relation
                self.infered_knowledge[relation.blocks[1]][relation.blocks[0]] = relation
                self.infer()
        
        # print("##################")
            
    def make_move(self, state : State, move : Union[Move, Relation], final_state : Union[State, None] = None, partner : Union['Player', None] = None):
        assert any(isinstance(move,t) for t in [Move, Relation])
        
        if isinstance(move, Move):
            state = self.make_physical_move(state, move, final_state, partner)
        else:
            self.share_knowledge(move)
        
        return state
    
    def make_copy(self):
        return Player(deepcopy(self.perspective), deepcopy(self.knowledge), deepcopy(self.num_blocks), deepcopy(self.shared_knowledge_flags), deepcopy(self.infered_knowledge), deepcopy(self.connected_relations_pairs))
    
    def __repr__(self):
        return f'Player({(self.perspective, self.knowledge, self.num_blocks, self.shared_knowledge_flags)} )'
    
    def init_infered_knowledge(self):
        """
        Inferred knowledge is a dictionary with keys as the blocks and values as the list of relations that can be inferred based on the current given knowledge
        It records all the possible pairs between blocks and blocks/bins. 
        The relationship between blocks and bins can be 1. Unknown (None) 2. Same 3. Different.
        The structure is: self.reference[block][bin] = RelationValue().
        The relationship between blocks and blocks can be 1. Unknown (NONE) 2. Relations()
        The structure is: self.reference[block][block] = Relations(). 
        
        """
        # initialize relations and groundings
        self.infered_knowledge = defaultdict(dict)
        blocks = [block for block in Block][:self.num_blocks]
        for block in blocks:
            for bin in [Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN, Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN]:
                self.infered_knowledge[block][bin] = Grounding(block, bin, RelationValue.NONE)
                self.infered_knowledge[bin][block] = Grounding(bin, block, RelationValue.NONE)
            
            for block2 in blocks:
                if block2 == block:
                    continue
                self.infered_knowledge[block][block2] = Relation(block, block2, RelationType.NONE, RelationValue.NONE)
        
        # update with known groundings and relations
        for grounding in self.knowledge.groundings:
            bin = grounding.bin
            block = grounding.block
            self.infered_knowledge[block][bin] = grounding
            self.infered_knowledge[bin][block] = grounding
        
        for relation in self.knowledge.relations:
            self.infered_knowledge[relation.blocks[0]][relation.blocks[1]] = relation
            self.infered_knowledge[relation.blocks[1]][relation.blocks[0]] = relation

        self.infer()
    
    def infer(self):
        """
        Create a graph and expand the knowledge based on the current knowledge
        nodes are blocks and bins
        edges are relations between blocks and blocks/bins
        """
        # define the nodes
        nodes = [block for block in Block][:self.num_blocks] + [Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN, Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN]
        # # define the edges
        # edges = []
        # for block in [block for block in Block][:self.num_blocks]:
        #     for bin in [Bins.TOP_LEFT_BIN, Bins.TOP_RIGHT_BIN, Bins.BOTTOM_LEFT_BIN, Bins.BOTTOM_RIGHT_BIN]:
        #         if self.infered_knowledge[block][bin] != RelationValue.NONE:
        #             # only add the known relations
        #             edges.append((block, bin, self.infered_knowledge[block][bin]))
        #             edges.append((bin, block, self.infered_knowledge[block][bin]))
        
        #     for block2 in [block for block in Block][:self.num_blocks]:
        #         if block2 == block:
        #             continue
        #         if self.infered_knowledge[block][block2].value != RelationValue.NONE:
        #             # only add the known relations
        #             edges.append((block, block2, self.infered_knowledge[block][block2]))
        
        def get_relation_value(nodeA, nodeB):
            if nodeA not in self.infered_knowledge:
                return RelationValue.NONE
            if nodeB not in self.infered_knowledge[nodeA]:
                return RelationValue.NONE
            rel = self.infered_knowledge[nodeA][nodeB].value
            return rel

        def update_relation(nodeA, nodeB, new_value):
            self.infered_knowledge[nodeA][nodeB] = new_value

        # Flag to keep track if any new inference was added in the current iteration.
        changed = True
        while changed:
            changed = False
            # Try all possible pairs A and B.
            for A in nodes:
                for B in nodes:
                    if A == B:
                        continue
                    # Skip if we already know the relation between A and B.
                    if get_relation_value(A, B) != RelationValue.NONE:
                        continue
                    # Look for an intermediary node C.
                    for C in nodes:
                        if C == A or C == B:
                            continue
                        value_AC = get_relation_value(A, C)
                        value_CB = get_relation_value(C, B)
                        # If both edges are known, we can try to infer A->B.
                        if value_AC != RelationValue.NONE and value_CB != RelationValue.NONE:
                            if (self.infered_knowledge[A][C], self.infered_knowledge[C][B]) in self.connected_relations_pairs:
                                # Have used this pair of knowledge to infer new knowledge
                                continue
                            # Assume combine_relation is your provided function.
                            inferred_value, X, Y = self.combine_relation(self.infered_knowledge[A][C], self.infered_knowledge[C][B])
                            self.connected_relations_pairs.add((self.infered_knowledge[A][C], self.infered_knowledge[C][B]))
                            if inferred_value != RelationValue.NONE and self.infered_knowledge[X][Y] != inferred_value:
                                # if self.perspective == Perspective.PLAYER_1:
                                #     print(f"Update {X} -> {Y} with {inferred_value}, previous value: {self.infered_knowledge[X][Y]}")
                                update_relation(X, Y, inferred_value)
                                update_relation(Y, X, inferred_value)
                                changed = True
                                # We found a new relation for A-B, so break out to restart the iteration.
                                break
                    # If any update was made, break out to restart scanning all node pairs.
                    if changed:
                        break
                if changed:
                    break
                    
    def display_infered_knowledge(self):
        print(f"######### Player {self.perspective.name} #########")
        ret_lst = []
        for node in self.infered_knowledge.keys():
            if isinstance(node, Block):
                block = node
                st = f'BLOCK-{block.value}: '
                for k, v in self.infered_knowledge[block].items():
                    st += f'{k}: {v}\n'
            else:
                bin = node
                st = f'BIN-{bin.name}: '
                for k, v in self.infered_knowledge[bin].items():    
                    st += f'{k}: {v}\n'
            ret_lst.append(st)
        return '\n'.join(ret_lst)
    
    def combine_relation(self, relation1, relation2):
        """
        Return the inferred relation based on the given relations.
        assumption: relation1 and relation2 have a block in common
        """
        # if self.perspective == Perspective.PLAYER_1:
        #     print(f"{relation1=}, {relation2=}")

        block1, block2, block3, bin, block4 = None, None, None, None, None
        if isinstance(relation1, Grounding) and isinstance(relation2, Grounding):
            return Relation(relation1.block, relation2.block, RelationType.BIN, RelationValue.SAME), relation1.block, relation2.block
        
        
        if isinstance(relation1, Grounding):
            block1, bin = relation1.block, relation1.bin
        elif isinstance(relation1, Relation):
            block1, block2 = relation1.blocks[0], relation1.blocks[1]
        
        if isinstance(relation2, Grounding):
            block3, bin = relation2.block, relation2.bin
        elif isinstance(relation2, Relation):
            block3, block4 = relation2.blocks[0], relation2.blocks[1]
        
        if block2 is None:
            # relation1 is a grounding, relation2 is a relation; Then we have 2 different blocks and 1 bin.
            block2 = block3 if block1 == block4 else block4
            row = bin.value[0].value
            col = bin.value[1].value
            if relation2.type == RelationType.BIN and relation2.value == RelationValue.SAME:
                return Grounding(block2, bin, RelationValue.SAME), block2, bin
            if relation2.type == RelationType.ROW and relation2.value == RelationValue.SAME:
                bin2 = Bins((Rows(row), Columns(1-col)))
                return Grounding(block2, bin2, RelationValue.SAME), block2, bin2
            if relation2.type == RelationType.COLUMN and relation2.value == RelationValue.SAME:
                bin2 = Bins((Rows(1-row), Columns(col)))
                return Grounding(block2, bin2, RelationValue.SAME), block2, bin2
            if relation2.type == RelationType.DIAGONAL and relation2.value == RelationValue.SAME:
                bin2 = Bins((Rows(1-row), Columns(1-col)))
                return Grounding(block2, bin2, RelationValue.SAME), block2, bin2
            raise ValueError(f"ERROR: {relation1} and {relation2}")
        
            
        if block4 is None:
            # relation 1 is a relation, relation 2 is a grounding; Then we have 2 different blocks and 1 bin.
            block4 = block1 if block2 == block3 else block2
            row = bin.value[0].value
            col = bin.value[1].value
            if relation1.type == RelationType.BIN and relation1.value == RelationValue.SAME:
                return Grounding(block4, bin, RelationValue.SAME), block4, bin
            if relation1.type == RelationType.ROW and relation1.value == RelationValue.SAME:
                bin2 = Bins((Rows(row), Columns(1-col)))
                return Grounding(block4, bin2, RelationValue.SAME), block4, bin2
            if relation1.type == RelationType.COLUMN and relation1.value == RelationValue.SAME:
                bin2 = Bins((Rows(1-row), Columns(col)))
                return Grounding(block4, bin2, RelationValue.SAME), block4, bin2
            if relation1.type == RelationType.DIAGONAL and relation1.value == RelationValue.SAME:
                bin2 = Bins((Rows(1-row), Columns(1-col)))
                return Grounding(block4, bin2, RelationValue.SAME), block4, bin2
            raise ValueError(f"ERROR: {relation1} and {relation2}")
        
        # both relations are relations
        # get the anchor block, suppose relation1 is A->B and relation2 is B->C
        A, B, C = None, None, None
        if block1 == block3:
            A, B, C = block2, block1, block4
        elif block1 == block4:
            A, B, C = block2, block1, block3
        elif block2 == block3:
            A, B, C = block1, block2, block4
        elif block2 == block4:
            A, B, C = block1, block2, block3
        if relation1.type == RelationType.BIN and relation1.value == RelationValue.SAME:
            if relation2.type == RelationType.BIN and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.BIN, RelationValue.SAME), A, C
            if relation2.type == RelationType.ROW and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.ROW, RelationValue.SAME), A, C
            if relation2.type == RelationType.COLUMN and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.COLUMN, RelationValue.SAME), A, C
            if relation2.type == RelationType.DIAGONAL and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.DIAGONAL, RelationValue.SAME), A, C
            raise ValueError(f"ERROR: {relation1} and {relation2}")
        
        if relation1.type == RelationType.ROW and relation1.value == RelationValue.SAME:
            if relation2.type == RelationType.BIN and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.ROW, RelationValue.SAME), A, C
            if relation2.type == RelationType.ROW and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.BIN, RelationValue.SAME), A, C
            if relation2.type == RelationType.COLUMN and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.DIAGONAL, RelationValue.SAME), A, C
            if relation2.type == RelationType.DIAGONAL and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.COLUMN, RelationValue.SAME), A, C
            raise ValueError(f"ERROR: {relation1} and {relation2}")
        
        if relation1.type == RelationType.COLUMN and relation1.value == RelationValue.SAME:
            if relation2.type == RelationType.BIN and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.COLUMN, RelationValue.SAME), A, C
            if relation2.type == RelationType.ROW and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.DIAGONAL, RelationValue.SAME), A, C
            if relation2.type == RelationType.COLUMN and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.BIN, RelationValue.SAME), A, C
            if relation2.type == RelationType.DIAGONAL and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.ROW, RelationValue.SAME), A, C
            raise ValueError(f"ERROR: {relation1} and {relation2}")

        if relation1.type == RelationType.DIAGONAL and relation1.value == RelationValue.SAME:
            if relation2.type == RelationType.BIN and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.DIAGONAL, RelationValue.SAME), A, C
            if relation2.type == RelationType.ROW and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.COLUMN, RelationValue.SAME), A, C
            if relation2.type == RelationType.COLUMN and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.ROW, RelationValue.SAME), A, C
            if relation2.type == RelationType.DIAGONAL and relation2.value == RelationValue.SAME:
                return Relation(A, C, RelationType.BIN, RelationValue.SAME), A, C
            raise ValueError(f"ERROR: {relation1} and {relation2}")
        
        raise ValueError(f"ERROR: {relation1} and {relation2}")
            
    def get_infered_knowledge(self):
        blocks = [block for block in Block][:self.num_blocks]
        ret_lst = set()
        for block in blocks:
            for k, v in self.infered_knowledge[block].items():
                if v.value != RelationValue.NONE:
                    ret_lst.add(v)
        
        ret_lst = list(ret_lst)
        groundings = [grounding for grounding in ret_lst if isinstance(grounding, Grounding)]
        relations = [relation for relation in ret_lst if isinstance(relation, Relation)]

        return Knowledge(relations, groundings)
        
        


            

            
            
        