from fastapi import FastAPI
from typing import List, Optional
from annotator import propose
from program import DafnyProgram
from search import SearchNode, VLLMProposer, batch_greedy_search
import os

NUM_PROPOSALS = int(os.environ.get("NUM_PROPOSALS", "2"))
LOCALIZED = os.environ.get("LOCALIZED", "false") != "false"
model_path = os.environ.get("MODEL_PATH", "gpoesia/dafny-annotator-8B")
tokenizer = os.environ.get("TOKENIZER", "meta-llama/Llama-3.1-8B")

proposer = VLLMProposer(model_path, tokenizer, num_proposals=NUM_PROPOSALS)

app = FastAPI()

@app.post("/annotate")
async def annotate(program: str) -> List[str]:
    node = SearchNode(DafnyProgram(program))
    results = proposer.propose([node])
    return results[0]

@app.post("/greedy_search")
async def search(program: str) -> Optional[str]:
    programs = [DafnyProgram(program)]
    results = batch_greedy_search(programs, proposer, 5, None, localized=LOCALIZED)
    result = results[0]
    return str(result) if result else None
