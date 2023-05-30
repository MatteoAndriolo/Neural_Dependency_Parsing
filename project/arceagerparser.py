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

LEFT_ARC = 0
RIGHT_ARC = 1
SHIFT = 2
REDUCE = 3


class ArcEager:
    def __init__(self, sentence, debug=False):
        self.sentence = sentence
        self.buffer = [i for i in range(len(self.sentence))]
        self.stack = []
        self.arcs = [-1 for _ in range(len(self.sentence))]
        self.debug = debug

        # three shift moves to initialize the stack
        self.shift()
        if self.debug:
            print("end configuration")

    def left_arc(self):
        s1 = self.stack.pop()
        b1 = self.buffer[0]
        self.arcs[s1] = b1
        if self.debug:
            print("left arc")
            self.print_configuration()

    def right_arc(self):
        s1 = self.stack[-1]
        b1 = self.buffer.pop(0)
        self.arcs[b1] = s1
        self.stack.append(b1)
        if self.debug:
            print("right arc")
            self.print_configuration()

    def shift(self):
        b1 = self.buffer.pop(0)
        self.stack.append(b1)
        if self.debug:
            print("shift")
            self.print_configuration()

    def reduce(self):
        self.stack.pop()
        if self.debug:
            print("reduce")
            self.print_configuration()

    def is_tree_final(self):
        return len(self.stack) == 1 and len(self.buffer) == 0

    def print_configuration(self):
        s = [self.sentence[i] for i in self.stack]
        b = [self.sentence[i] for i in self.buffer]
        print(s, b)
        print(self.stack, self.buffer)
        print(self.arcs)


class Oracle:
    def __init__(self, parser, gold_tree):
        self.parser = parser
        self.gold = gold_tree

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
            len(self.parser.stack) == 0  # never happens
            or len(self.parser.buffer) == 0  # no left arc to produce
            or len(self.parser.stack) == 1  # should be != 1
            or self.parser.stack[-1] == 0  # if top is ROOT
        ):
            return False

        s = self.parser.stack[-1]
        b = self.parser.buffer[0]  # [0]
        if self.gold[s] == b:
            return True

        return False

    def is_right_arc_gold(self):
        # if topmost stack element is gold head of the first element of the buffer
        if len(self.parser.stack) == 0 or len(self.parser.buffer) == 0:
            return False

        s = self.parser.stack[-1]
        b = self.parser.buffer[0]  # [0]

        if self.gold[b] == s:
            return True

        return False

    def is_reduce_gold(self):
        # stack empty  || top no head --> return False
        if len(self.parser.stack) == 1:
            return False
        if len(self.parser.buffer) == 0:
            if self.parser.arcs[self.parser.stack[-1]] != -1:
                return True
        for i in len(self.parser.buffer):
            if self.parser.stack[-1] == self.parser.arcs[i]:
                return False
            return False
        # if exist a link k <-/-> next, k < top then return REDUCE
        s = self.parser.stack[-1]
        for i in range(0, len(self.parser.buffer)):
            b = self.parser.buffer[i]

            if self.gold[b] == s or self.gold[s] == b:
                return True
        return False

    def is_shift_gold(self):
        if len(self.parser.buffer) == 0:
            return False

        # This dictates transition precedence of the parser
        if self.is_left_arc_gold() or self.is_right_arc_gold() or self.is_reduce_gold():
            return False

        return True


if __name__ == "__main__":
    from datasets import load_dataset

    sentence = ["<ROOT>", "Their", "point", "has", "been", "made", "."]
    gold = [-1, 2, 5, 5, 5, 0, 5]
    # sentence = ["<ROOT>", "He", "began", "to", "write", "again", "."]
    # gold = [-1, 2, 0, 4, 2, 4, 2]
    sentence = ["<ROOT>", "Chart", "with", "category", "field"]
    gold = [-1, 0, 4, 4, 1]
    parser = ArcEager(sentence)
    oracle = Oracle(parser, gold)
    c = 0
    while not parser.is_tree_final():
        print(c)
        c += 1
        if oracle.is_left_arc_gold():
            parser.left_arc()
        elif oracle.is_right_arc_gold():
            parser.right_arc()
        elif oracle.is_reduce_gold():
            parser.reduce()
        elif oracle.is_shift_gold():
            parser.shift()
        else:
            print("error")
            break
        parser.print_configuration()
        print(gold)
        input()
