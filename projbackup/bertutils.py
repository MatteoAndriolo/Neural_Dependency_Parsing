from typing import List


def match_subtokens(l1:List[str], l2:List[str]):
    # Create output list
    output:List[List[int]] = []
    # Initialize index for l2
    index = 0
    # Iterate through l1
    for token in l1:
        subtoken_indices = []
        # Get the indices of the subtokens
        while index < len(l2) and (not subtoken_indices or l2[index].startswith("#")):
            subtoken_indices.append(index)
            index += 1
        # Append subtoken indices to output
        output.append(subtoken_indices)
    return output


  
def tokens_tokenizer_correspondence(tokens:List[List[str]], berttokens:List[List[int]]):
    global tokenizer
    correspondences:List[List[List[int]]]=[]
    
    for t,bt in zip(tokens, berttokens):
        correspondences.append(match_subtokens(t, list(tokenizer.convert_ids_to_tokens(bt))))
    return correspondences
            
    git filter-branch --prune-empty -d project/ --index-filter "git rm --cached -f --ignore-unmatch model_e*"  --tag-name-filter cat -- --all