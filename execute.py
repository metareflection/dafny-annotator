import hashlib
import os

def execute(cmd, ext, v):
    HOME = os.environ["HOME"]
    TMP_DIR = f"{HOME}/tmp/dafny-annotator/{ext}/"
    key = hashlib.md5(v.encode("utf-8")).hexdigest()
    dir = "%s%s/" % (TMP_DIR, key)
    old_dir = os.getcwd()
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.chdir(dir)

    try:
        fn = f"ex.{ext}"
        outfn = "out.txt"
        errfn = "err.txt"

        f = open(fn, "w", encoding='utf-8')
        f.write(v)
        f.close()

        status = os.system("timeout 10s %s %s >%s 2>%s" % (cmd, fn, outfn, errfn))

        f = open(outfn, "r", encoding='utf-8')
        outlog = f.read()
        f.close()

        f = open(errfn, "r", encoding='utf-8')
        log = f.read()
        f.close()

        sys_error_prefix = "sh: line 1:"
        if log.startswith(sys_error_prefix):
            raise RuntimeError(
                log[len(sys_error_prefix) :]
                + " -- install tool locally"
            )
    finally:
        os.chdir(old_dir)

    return {"status": status, "log": log, "out": outlog}
