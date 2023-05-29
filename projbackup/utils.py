from typing import List
from arceagerparser import ArcEager, Oracle

## -------- General functions ----------------
def is_projective(tree):
    for i in range(len(tree)):
        if tree[i] == -1:
            continue
        left = min(i, tree[i])
        right = max(i, tree[i])

        for j in range(0, left):
            if tree[j] > left and tree[j] < right:
                return False
        for j in range(left + 1, right):
            if tree[j] < left or tree[j] > right:
                return False
        for j in range(right + 1, len(tree)):
            if tree[j] > left and tree[j] < right:
                return False

    return True


def generate_gold_path(sentence:List[str], gold:List[int]) -> tuple[List[List[int]], List[int]]:
  '''
  sentence: list of tokens
  gold: list of heads
  '''
  parser = ArcEager(sentence)
  oracle = Oracle(parser, gold)

  gold_configurations:List[List[int]] = []
  gold_moves:List[int] = []

  while not parser.is_tree_final():
      # save configuration - index of token in sentence
      configuration = [
          parser.stack[ - 1],
      ]
      if len(parser.buffer) == 0:
          configuration.append(-1)
      else:
          configuration.append(parser.buffer[0])
      
      # save configuration    
      gold_configurations.append(configuration)

          # save gold move
      if oracle.is_left_arc_gold():
          gold_moves.append(0)
          parser.left_arc()
      elif oracle.is_right_arc_gold():
          gold_moves.append(1)
          parser.right_arc()
      elif oracle.is_shift_gold():
          gold_moves.append(2)
          parser.shift()
      elif oracle.is_reduce_gold():
          gold_moves.append(3)
          parser.reduce()
  
  return gold_configurations, gold_moves, 

## ----------- BiLSTM ----------------------------------



# ----------------- BERT ------------------------------------------------