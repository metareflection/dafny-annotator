from transformers import AutoModelForCausalLM
#from peft import AutoPeftModelForCausalLM
from fastapi import FastAPI, Query
from typing import List, Optional
from annotator import propose
from program import DafnyProgram

model = AutoModelForCausalLM.from_pretrained("gpoesia/dafny-annotator-8B", device_map="auto")
#model = AutoPeftModelForCausalLM.from_pretrained("gpoesia/finetuned_Meta-Llama-3.1-8B_dafnybench-100-peft", device_map="auto")

app = FastAPI()

@app.post("/annotate")
async def annotate(program: str, num_samples: Optional[int] = Query(None, description="Number of samples (default to 1)")) -> List[str]:
    r = propose(model, DafnyProgram(program), num_samples or 1)
    return [str(p) for p in r]
