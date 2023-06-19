from typing import List

def is_projective(head):
    for i in range(len(head)):
        if head[i] == -1:
            continue
        left = min(i, head[i])
        right = max(i, head[i])

        for j in range(0, left):
            if head[j] > left and head[j] < right:
                return False
        for j in range(left + 1, right):
            if head[j] < left or head[j] > right:
                return False
        for j in range(right + 1, len(head)):
            if head[j] > left and head[j] < right:
                return False

    return True



from arceagerparser import LEFT_ARC,RIGHT_ARC,REDUCE,SHIFT, NOMOVE,ArcEager, Oracle, generate_gold


def evaluate(gold:List[List[int]], preds:List[List[int]]):
    total = 0
    correct = 0

    for g, p in zip(gold, preds):
        for i in range(1, len(g)):
            total += 1
            if g[i] == p[i]:
                correct += 1

    return correct / total

from arceagerparser import is_left_possible, is_right_possible, is_reduce_possible, is_shift_possible
from torch import sort as tsort, Tensor

def parse_step(parsers:List[ArcEager], moves:Tensor):
    _, indices = tsort(moves, descending=True)
    list_moves=[]
    for i in range(len(parsers)):
        noMove =True 
        if parsers[i].is_tree_final():
           list_moves.append(NOMOVE) 
        for j in range(4):
            if indices[i][j] == LEFT_ARC and is_left_possible(parsers[i]):
                list_moves.append(LEFT_ARC)
                noMove = False;break;
            elif indices[i][j] == RIGHT_ARC and is_right_possible(parsers[i]):
                list_moves.append(RIGHT_ARC)
                noMove = False;break;
            elif indices[i][j] == REDUCE and is_reduce_possible(parsers[i]):
                list_moves.append(REDUCE)
                noMove = False;break;
            elif indices[i][j] == SHIFT and is_shift_possible(parsers[i]) :
                list_moves.append(SHIFT)
                noMove = False;break;
        if noMove:
            list_moves.append(NOMOVE)
    return list_moves
                

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
                                
                                
if __name__== "__main__":
    from datasets import load_dataset
    from utils import is_projective
    from arceagerparser import generate_moves_configurations_heads 
    training_dataset=load_dataset("universal_dependencies", "en_lines", split="train")
    training_dataset=training_dataset.filter(lambda x: is_projective([-1]+list(map(int,x["head"]))))
    errors=False
    
    for i,a in enumerate(training_dataset):
        tokens=["<ROOT>"]+a["tokens"]
        heads=[-1]+list(map(int,a["head"]))
        
        parser = ArcEager(tokens)
        oracle = Oracle(parser, heads)
        
        moves,_,_  =generate_moves_configurations_heads(parser, oracle)
        _, moves2= generate_gold_pathmoves(tokens, heads)

        if moves!=moves2:
            errors=True
            
    if errors:
        print("ERROR FOUND")
        print("TEST FAILED")
    else:
        print("TEST PASSED")