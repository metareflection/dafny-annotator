#from transformers import AutoTokenizer, AutoModelForCausalLM
#from peft import AutoPeftModelForCausalLM
from fastapi import FastAPI, Query
from typing import List, Optional
from annotator import propose
from program import DafnyProgram
from search import SearchNode, VLLMProposer

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
#model = AutoModelForCausalLM.from_pretrained("gpoesia/dafny-annotator-8B", device_map="auto")
#model = AutoPeftModelForCausalLM.from_pretrained("gpoesia/finetuned_Meta-Llama-3.1-8B_dafnybench-100-peft", device_map="auto")
proposer = VLLMProposer("gpoesia/dafny-annotator-8B", "meta-llama/Llama-3.1-8B")

app = FastAPI()

@app.post("/annotate")
#async def annotate(program: str, num_samples: Optional[int] = Query(None, description="Number of samples (default to 1)")) -> List[str]:
async def annotate(program: str) -> List[str]:
    node = SearchNode(DafnyProgram(program))
    result = proposer.propose([node])
    return result[0]
