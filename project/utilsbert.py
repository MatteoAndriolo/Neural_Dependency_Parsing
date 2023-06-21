
from transformers import AutoTokenizer
def match_subtokens(l1:list[str], l2:list[str]):
    # Create output list
    output:list[list[int]] = []
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

def merge_splitted_tokens(tokens:list[str]):
    out:list[str]=[]
    len=0
    for t in tokens:
      if t[0]=="#":
        out[len-1]+=t.strip("#")
      else:
        out.append(t)
        len+=1
    return out


def tokens_tokenizer_correspondence(tokens:list[list[str]], berttokens:list[list[int]], tokenizer:AutoTokenizer):
    correspondences:list[list[list[int]]]=[]
    
    for t,bt in zip(tokens, berttokens):
        corr=match_subtokens(t, list(tokenizer.convert_ids_to_tokens(bt)))
        correspondences.append(corr)
        
        
    return correspondences

    
if __name__=="__main__":
    from datasets import load_dataset
    from utils import is_projective
    from arceagerparser import generate_gold 
    training_dataset=load_dataset("universal_dependencies", "en_lines", split="train")
    training_dataset=training_dataset.filter(lambda x: is_projective([-1]+list(map(int,x["head"]))))
    errors=False
    
    tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(["<ROOT>", "<EMPTY>"], special_tokens=True)
    
    tokenized_sentences= tokenizer(["<ROOT> "+td["text"] for td in training_dataset], add_special_tokens=False)

    
    correspondences= tokens_tokenizer_correspondence([["<ROOT>"]+td for td in training_dataset["tokens"]], tokenized_sentences["input_ids"], tokenizer) #type:ignore


    
    if errors:
        print("ERROR FOUND")
        print("TEST FAILED")
    else:
        print("TEST PASSED")