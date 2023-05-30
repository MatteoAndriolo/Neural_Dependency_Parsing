from arceagerparser import ArcEager, Oracle
from typing import List

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



from arceagerparser import LEFT_ARC,RIGHT_ARC,REDUCE,SHIFT
def generate_gold_pathmoves(sentence:List[str], gold:List[int]) -> tuple[List[List[int]], List[int]]:
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
          gold_moves.append(LEFT_ARC)
          parser.left_arc()
      elif oracle.is_right_arc_gold():
          gold_moves.append(RIGHT_ARC)
          parser.right_arc()
      elif oracle.is_shift_gold():
          gold_moves.append(SHIFT)
          parser.shift()
      elif oracle.is_reduce_gold():
          gold_moves.append(REDUCE)
          parser.reduce()
      
  
  return gold_configurations, gold_moves



def evaluate(gold:List[List[int]], preds:List[List[int]]):
    total = 0
    correct = 0

    for g, p in zip(gold, preds):
        for i in range(1, len(g)):
            total += 1
            if g[i] == p[i]:
                correct += 1

    return correct / total


from torch import sort as tsort, Tensor
def parse_step(parsers:List[ArcEager], moves:Tensor):
    _, indices = tsort(moves, descending=True)
    
    for i in range(len(parsers)):
        noMove =True 
        # Conditions
        cond_left = (
            len(parsers[i].stack)>0
            and len(parsers[i].buffer)>0
            and parsers[i].stack[-1] != 0
        )
        cond_right = len(parsers[i].stack)>0 and len(parsers[i].buffer) >0
        cond_reduce = len(parsers[i].stack) and parsers[i].stack[-1] != 0
        cond_shift = len(parsers[i].buffer) > 0

        for j in range(4):
            if parsers[i].is_tree_final():
                noMove = False;break;
            else:
                if indices[i][j] == LEFT_ARC and cond_left:
                    #print("left")
                    parsers[i].left_arc()
                    noMove = False;break;
                elif indices[i][j] == RIGHT_ARC and cond_right:
                    #print("right")
                    parsers[i].right_arc()
                    noMove = False;break;
                elif indices[i][j] == SHIFT and cond_shift:
                    #print("shift")
                    parsers[i].shift()
                    noMove = False;break;
                elif indices[i][j] == REDUCE and cond_reduce:
                    #print("reduce")
                    parsers[i].reduce()
                    noMove = False;break;
        if noMove:
            print(parsers[i].stack, parsers[i].buffer)
            print("noMove was possible")
            exit(-5)
            
                

# In this function we select and perform the next move according to the scores obtained.
# We need to be careful and select correct moves, e.g. don't do a shift if the buffer
# is empty or a left arc if σ2 is the ROOT. For clarity sake we didn't implement
# these checks in the parser so we must do them here. This renders the function quite ugly
# 0 Lx; 1 Rx, 2 shifr; 3 reduce
def parse_step_2(parsers: List[ArcEager], moves:List[List[int]]):
    moves_argm = moves.argmax(-1)
    for i in range(len(parsers)):
        noMove = False
        # Conditions
        cond_left = (
            len(parsers[i].stack)>0
            and len(parsers[i].buffer)>0
            and parsers[i].stack[-1] != 0
        )
        cond_right = len(parsers[i].stack)>0 and len(parsers[i].buffer)>0
        cond_reduce = len(parsers[i].stack)>0 and parsers[i].stack[-1] != 0
        cond_shift = len(parsers[i].buffer) > 0
        if parsers[i].is_tree_final():
            continue
        else:
            if moves_argm[i] == LEFT_ARC:
#------------------------------ firdt condition to check is the left arc -> right arc -> shift -> reduce------------------------------
                if cond_left:
                    parsers[i].left_arc()
                else:
                    if cond_right:
                        parsers[i].right_arc()
                    elif cond_reduce:
                        parsers[i].reduce()
                    elif cond_shift:
                        parsers[i].shift()
                    else:
                        print("noMove was possible on left")
#------------------------------ firdt condition to check is the right arc -> shift -> reduce------------------------------
            if moves_argm[i] == RIGHT_ARC:
                #print("right")
                if cond_right:
                    parsers[i].right_arc()
                else:
                    if cond_reduce:
                        parsers[i].reduce() 
                    elif cond_shift:
                        parsers[i].shift()
                    else:
                        print("noMove was possible on right")
#------------------------------ firdt condition to check is the shift -> reduce------------------------------
            if moves_argm[i] == SHIFT:
                if cond_shift:
                    parsers[i].shift()
                elif cond_reduce:
                    parsers[i].reduce()
                else:
                    print("noMove was possible on shift")
#------------------------------ firdt condition to check is the reduce and if no reduce was possible take in account the probabilities ------------------------------
            if moves_argm[i] == REDUCE:
                if cond_reduce:
                    parsers[i].reduce()
                else:
                    if moves[i][0] > moves[i][1] and moves[i][0] > moves[i][2] and cond_left:
                        parsers[i].left_arc()
                    else:
                        if moves[i][1] > moves[i][2] and cond_right:
                            parsers[i].right_arc()
                        else:
                            if cond_shift:
                                parsers[i].shift()
                            else:
                                print(moves[i][0], moves[i][1], moves[i][2], cond_left, cond_right, cond_shift)