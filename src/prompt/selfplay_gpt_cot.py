SYSTEM_PROMPT = """You are playing a cooperative game where you and another player must sort blocks into the correct bins as quickly as possible. Each player has knowledge about the expected placement of the blocks, such as whether blocks should be aligned in the same row, column, or diagonal. You can only move blocks into bins near you or into a shared bin accessible to both players. You cannot access the other player's bins.

The game concludes when all blocks are correctly placed in the bins. During your turn, you have several options:
- Move a block from a bin to another bin. Both bins must be accessible to you.
- Share a piece of knowledge with the other player. The shared knowledge should be selected from the knowledge you have. You cannot share knowledge you do not have.
- Request knowledge from the other player about a specific block. 
- Pass your turn.

## General Guidance
- You can only move blocks to bins accessible to you.
- You can only share knowledge you have. DO NOT share knowledge you do not have, nor request knowledge you already have.
- You should request knowledge about a block that you do not have knowledge of.
- You can only move one block at a time.
- When the knowledge says two blocks are in the same row, column, or diagonal, it means they are not in the same bin.
- If you or the other player make an incorrect move, it will tell you in the action history. You MUST NOT make the same incorrect move again. You MUST carefully check the action history before you make a move.
- If your partner ask about the knowledge of a block, you should provide the knowledge if you have it. Carefully think about which piece of knowledge is the most helpful to share.
- The game will stop players from placing the blocks into the wrong bins. So if you see a block placed in a bin, that means it is the correct bin for that block.


## Reasoning Format
You need to reason about which action to take before you make the decision. Some examples of reasoning:
- "According to my knowledge, block0 should be in top-right bin. Since I cannot reach the top-right bin, I should pass it to the common bin so that my partner can take it."
- "I know block1 and block2 should be in the same row. I also know block1 should be placed in the top-right bin. So I should move block2 to the top-left bin, which is also in the same row with the top-right bin."
- "Block1 is on the same row with block2. Block2 is on the same column with block3. So I can deduce that block1 and block3 should be on the same diagonal."
- "Block1 is still not moved but I cannot reach it. I should share my knowledge about block1 with my partner so that it can move it to the correct position."
- "All the blocks are in the correct position except block2. However, according to my knowledge I don't know where the block2 should go in. I may ask my partner about the knowledge of it."
- "I have no knowledge about block2, but I saw block2 is already placed in top-left bin, so I should assume that it is the correct final location."

## Action Format
Your actions must be formatted as follows:
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

Please strictly follow the format above to ensure the game runs smoothly.

## Output Format
Please provide your output in this format:
<THINK><your reasoning></THINK><ACTION><your action></ACTION>

"""

USER_PROMPT = """You are Player{player_id} on the {side} side of the game board. You have the following knowledge for your goal:
{knowledge}
Currently, the blocks are located as follows:
{blocks}
You can access these bins:
{bins}

The history of the game till now:
{move_history}

What action would you like to take? You need to be fully convinced of your action before you make a move. Please provide your reasoning before your action. Your reasoning should be concise enough within 3 sentences.
"""
