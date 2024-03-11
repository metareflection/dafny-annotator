import outlines

from cmdline import args
from test_example import program, verification_prompt

model = outlines.models.transformers(args.model, model_kwargs={'load_in_8bit':True})

#ln = outlines.generate.format(model, int)
program_lines = program.split('\n')
all_program_lines = map(str, range(1, len(program_lines)+1))
ln = outlines.generate.choice(model, all_program_lines)
cmd = outlines.generate.choice(model, ["assert", "invariant"])
prop = outlines.generate.regex(model, r"[^;]+;")

def gen1():
    p1 = f"{verification_prompt}\n// On line "
    r1 = ln(p1, max_tokens=3)
    #print(r1)
    p2 = f"{p1}, add "
    r2 = cmd(p2)
    #print(r2)
    p3 = f"{p2} {r2} "
    r3 = prop(p3, max_tokens=100)
    #print(r3)
    return (r1, r2, r3)

for i in range(0, 10):
    print(gen1())
