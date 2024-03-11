from execute import execute

def check(v: str) -> dict:
    return execute("dafny verify", "dfy", v)

def run(v: str) -> dict:
    return execute("dafny run", "dfy", v)
