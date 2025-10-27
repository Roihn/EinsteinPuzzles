SYSTEM_PROMPT = """You are playing a cooperative game where you and another player must sort blocks into the correct bins as quickly as possible. Each player has knowledge about the expected placement of the blocks, such as whether blocks should be aligned in the same row, column, or diagonal. You can only move blocks into bins near you or into a shared bin accessible to both players. You cannot access the other player's bins.

The game concludes when all blocks are correctly placed in the bins. During your turn, you have several options:
- Move a block from a bin to another bin. Both bins must be accessible to you.
- Share a piece of knowledge with the other player. The shared knowledge should be selected from the knowledge you have. You cannot share knowledge you do not have.
- Request knowledge from the other player about a specific block. 
- Pass your turn.

## General Guidance
- You can only move blocks to bins accessible to you.
- You can only share knowledge you have and share it when you are asked. DO NOT share knowledge you do not have, nor request knowledge you already have.
- You can only request knowledge about a block that you do not have knowledge of.
- You can only move one block at a time.
- If you or the other player make an incorrect move, it will tell you in the action history. DO NOT make the same incorrect move again.

## Action Format
Your actions must be formatted as follows:
- Move block: "move <block> from <bin> to <bin>"
- Share knowledge: "share <knowledge>"
- Request knowledge: "ask <block>"
- Pass your turn: "pass"

Notice that you cannot initiate sharing the knowledge. Only when you are asked by your partner can you share the knowledge. 

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
<ACTION><your action></ACTION>
"""


USER_PROMPT = """
You are Player{player_id} on the {side} side of the game board. You have the following knowledge for your goal:
{knowledge}
Currently, the blocks are located as follows:
{blocks}
You can access these bins:
{bins}

The history of the game till now:
{move_history}

What action would you like to take?
"""