"""
There are four possible transitions from a configuration where top is the word on top of the stack (if any) and next is the first word of the buffer:
0. Left-Arc adds a dependency arc from next to top and pops the stack;
1. Right-Arc adds a dependency arc from top to next and moves next to the
2. Reduce pops the stack; allowed only if top has a head.
3. Shift moves next to the stack.
stack.
allowed only if top has no head.

> left dependents are added bottom–up and right dependents top–down


However, a fundamental problem 23
with this system is that it does not guarantee that the output parse is a projective
dependency tree, only a projective dependency forest, that is, 
> a sequence of adjacent non-overlapping projective trees (Nivre 2008).
The failure to implement the tree constraint may lead to fragmented parses and lower parsing accuracy, especially with respect to the global structure of the sentence.
Moreover, even if the loss in accuracy is not substantial, this may be problematic when using the parser in applications where downstream components may not function correctly if the parser output is not a well-formed tree.

The standard solution to this problem in practical implementations, such as Malt-
Parser (Nivre, Hall, and Nilsson 2006), is to use an artificial root node and to attach
all remaining words on the stack to the root node at the end of parsing. This fixes the
formal problem, but normally does not improve accuracy because it is usually unlikely
that more than one word should attach to the artificial root node. Thus, in the error
analysis presented by McDonald and Nivre (2007), MaltParser tends to have very low
precision on attachments to the root node.

Other heuristic solutions have been tried,
usually by post-processing the nodes remaining on the stack in some way, but these
techniques often require modifications to the training procedure and/or undermine the
linear time complexity of the parsing system.
In any case, a clean theoretical solution to
this problem has so far been lacking.
"""
from typing import List


NOMOVE = -1
LEFT_ARC = 0
RIGHT_ARC = 1
REDUCE = 2
SHIFT = 3

IS_FINAL = -10

EMPTY = -1

class ArcEager:
    def __init__(self, sentence):
        """
        input:
            sentence: list of words | first word must be <ROOT>
            debug: if True print each move
        """
        if all([isinstance(x,str) for x in sentence]):
            if sentence[0] != "<ROOT>":
                raise Exception("ERROR: first word must be <ROOT>")
        elif all([isinstance(x,int) for x in sentence]):
            if sentence[0] != 1: # token of ROOT is 1
                raise Exception("ERROR: first word must be -1")
        else:
            raise Exception("ERROR: sentence must be list of words or list of ints")
            
        self.sentence = sentence
        self.buffer = [i for i in range(len(self.sentence))]
        self.stack = []

        self.list_arcs = [-1 for _ in range(len(self.sentence))]
        self.list_moves=[]
        self.list_configurations = []

        # Do first shift -> add ROOT to stack
        self.stack.append(self.buffer.pop(0))
        self.is_finished=False

    def update_configurations(self, move):
        ''' to do before each move '''
        if move == NOMOVE:
            self.list_configurations.append([EMPTY, EMPTY])
            self.list_moves.append(NOMOVE)
        if len(self.stack)>0:
            self.list_configurations.append([
                self.stack[-1],
                self.buffer[0] if len(self.buffer)>0 else EMPTY
            ])
            self.list_moves.append(move)
            
        
    def left_arc(self):
        self.update_configurations(LEFT_ARC)
        s1 = self.stack.pop(-1)
        b1 = self.buffer[0]
        self.list_arcs[s1] = b1

    def right_arc(self):
        if not is_right_possible(self):
            self.nomove()
            return
        self.update_configurations(RIGHT_ARC)
        s1 = self.stack[-1]
        b1 = self.buffer.pop(0)
        self.stack.append(b1)
        self.list_arcs[b1] = s1

    def shift(self):
        self.update_configurations(SHIFT)
        self.stack.append(self.buffer.pop(0))

    def reduce(self):
        self.update_configurations(REDUCE)
        self.stack.pop()

    def nomove(self):
        self.is_finished=True
        self.update_configurations(NOMOVE)

    def do_move(self, move:int):
        if move==LEFT_ARC: 
            self.left_arc()
        elif move==RIGHT_ARC:
            self.right_arc()
        elif move==SHIFT:
            self.shift()
        elif move==REDUCE:
            self.reduce()
        elif move==NOMOVE:
            self.nomove()
        return move

    def is_tree_final(self):
        return self.is_finished or (len(self.stack) == 1 and len(self.buffer) == 0) 

    def print_configuration(self):
        s = [self.sentence[i] for i in self.stack]
        b = [self.sentence[i] for i in self.buffer]
        print(s, b)
        print(self.stack, self.buffer)
        print(self.list_arcs)

    def get_list_moves(self):
        return self.list_moves

    def get_list_configurations(self):
        return self.list_configurations

    def get_list_arcs(self):
        return self.list_arcs
    
    def get_configuration_now(self):
        if self.is_tree_final():
            conf=[-1,-1]
        else:
            conf=[self.stack[-1]]
            if len(self.buffer) == 0:
                conf.append(-1)
            else:
                conf.append(self.buffer[0])
        return conf
        
######################################################################################################
class Oracle:
    def __init__(self, parser, gold_tree:List[int]):
        self.parser = parser
        self.gold = list(map(int,gold_tree))

        # Check correctness of input
        if self.gold[0] != -1:
            print("ERROR: gold tree must start with -1")
            exit(-1)

    """
    i: top of stack, j: top of buffer
    if there's a link j -> i then return LEFT-ARC
    else if there's a link i -> j then return RIGHT-ARC
    else if there's a link k <-/-> j, k < i then return REDUCE
    else return SHIFT 
    """
    def is_left_arc_gold(self):
        # first element of the of the buffer is the gold head of the topmost element of the stack
        # if empty lists or if top has no head -> return False
        if (
            len(self.parser.buffer) == 0
            or self.parser.stack[-1] == 0  # if top is ROOT
        ):
            return False

        s = self.parser.stack[-1]
        b = self.parser.buffer[0]  # [0]
        if self.gold[s] != b:
            return False

        return True

    def is_right_arc_gold(self):
        # if topmost stack element is gold head of the first element of the buffer
        if len(self.parser.buffer) == 0:
            return False

        s = self.parser.stack[-1]
        b = self.parser.buffer[0]  # [0]
        if self.gold[b] != s:
            return False 

        return True 

    def is_reduce_gold(self):
        s = self.parser.stack[-1]
        if self.parser.list_arcs[s] == -1 or s==0: # if top has no head or if top is ROOT
            return False
        if len(self.parser.buffer) == 0:                    # if buffer is empty
            if self.parser.list_arcs[s] != -1 and s != 0:   # if top has a head and top is not ROOT
                return True
            return False

        for i in range(0, len(self.parser.buffer)):
            b = self.parser.buffer[i]
            if self.gold[b] == s or self.gold[s] == b: # if there's a link k <-/-> j, k < i then do not reduce
                return False 
            
        return True 

    def is_shift_gold(self):
        if len(self.parser.buffer) == 0:
            return False
        if self.is_left_arc_gold() or self.is_right_arc_gold() or self.is_reduce_gold():
            return False
        return True
    
    def get_next_move(self, do_it=False):
        if self.parser.is_tree_final():
            return IS_FINAL
        if self.is_left_arc_gold():
            return LEFT_ARC
        elif self.is_right_arc_gold():
            return RIGHT_ARC
        elif self.is_reduce_gold():
            return REDUCE
        elif self.is_shift_gold():
            return SHIFT
        else:
            print("NO MOVE")
            print(self.gold)
            print(self.parser.list_arcs)
            self.parser.print_configuration()
            exit(-5)
            return None

def is_left_possible(parser):
    return len(parser.stack) >= 1 and len(parser.buffer) >= 1 and parser.stack[-1] != 0

def is_right_possible(parser):
    return len(parser.stack) >= 1 and len(parser.buffer) >= 1

def is_shift_possible(parser):
    return len(parser.buffer) >= 1

def is_reduce_possible(parser):
    return len(parser.stack) >= 1 and parser.list_arcs[parser.stack[-1]] != -1

############################################################################################################
def generate_gold(sentence:List[str], gold:List[int]):
    '''
    Generate moves configurations heads for a given parser and oracle
    
    input:
        parser: ArcEager object
        oracle: Oracle object
    returns:
        moves: list of moves
        configurations: list of configurations
        arcs: list of heads
        
    '''
    parser:ArcEager=ArcEager(sentence)
    oracle:Oracle=Oracle(parser, gold)

    while not parser.is_tree_final():
        if parser.do_move(oracle.get_next_move()) == NOMOVE:
            print("ERROR: NOMOVE")
        
    return parser.list_moves, parser.list_configurations,  parser.list_arcs

############################################################################################################
if __name__ == "__main__":
    from datasets import load_dataset
    from utils import is_projective
    errors=False
    training_dataset=load_dataset("universal_dependencies", "en_lines", split="train")
    training_dataset=training_dataset.filter(lambda x: is_projective([-1]+list(map(int,x["head"]))))
    
    for i,a in enumerate(training_dataset):
        tokens=["<ROOT>"]+a["tokens"]
        heads=[-1]+list(map(int,a["head"]))
        print(tokens)
        print(heads)
        _, _, arcs = generate_gold(tokens, heads)
        print(arcs)
        if arcs != heads:
            print(f"ERROR HEADS in {i} sentence")
            errors=True
            break
    
    if errors:
        print("ERRORS FOUND")
        print("TEST NOT PASSED")
    else:
        print("TEST PASSED")
        

        
    
    
    
    