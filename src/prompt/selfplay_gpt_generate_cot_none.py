SYSTEM_PROMPT = """You are the assistant that provides the reasoning process for the given plays of one player in the game.

You are watching an agent playing a cooperative game where two players must sort blocks into the correct bins as quickly as possible. Each player has knowledge about the expected placement of the blocks, such as whether blocks should be aligned in the same row, column, or diagonal. It can only move blocks into bins near it or into a shared bin accessible to both players. It cannot access the other player's bins.

The game finishes when all blocks are correctly placed in the required bins. During the agent's turn, it has several options:
- Move a block to a nearby bin or the shared bin.
- Pass its turn.

Its actions must be formatted as follows:
- Move block: "move <block> from <bin> to <bin>"
- Pass its turn: "pass"

## Your Task
You are given the agent's action and you need to provide the reasoning behind the action. Please provide your output in first-person view as if you are the agent that makes the decision.

Some examples:
- "According to my knowledge, block0 should be in top-right bin. Since I cannot reach the top-right bin, I should pass it to the common bin so that my partner can take it."
- "I know block1 and block2 should be in the same row. I also know block1 should be placed in the top-right bin. So I should move block2 to the top-left bin, which is also in the same row with the top-right bin."
- "Block1 is not in the correct position according to my knowledge but I cannot reach it. I should wait for my partner to put it into the common bin so that I can reach it."
- "All the blocks are in the correct position except block2. However, according to my knowledge I don't know where the block2 should go in. I may randomly try to place it to one of the bins in front of me."
- "I can reach Block1 and Block3 since they are in front of me. I know they should be in the same row, but I do not know either of their exact expected locations. I will try with moving Block1 to the top-left bin as a start, supposing they are both on the top row."


## General Guidelines

- Provide a clear and concise explanation for the agent's action. No more than 2-3 sentences are needed.
- If you are not sure about the reasoning, this may because the given action is a random guess, and it is correct by luck. In this case, you should reason about what you know (briefly) and don't know (important), and clarify that the action is a guess.
- The given action may not be the best move, but you should explain the reasoning behind it.
- You can refer to the agent as "I" or "me" in your response.
"""

USER_PROMPT = """

Player{player_id} is on the {side} side of the game board. It has the following information for its goal:
{knowledge}
Currently, the blocks are located as follows:
{blocks}
It can access these bins:
{bins}

The history of the game till now:
{move_history}

It takes the action: {cur_move}
What is the reasoning behind this action? Please provide your thoughts as if you are the player.
"""