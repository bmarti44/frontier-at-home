import ctypes, hashlib, json, multiprocessing, os, pathlib, sqlite3, ssl, sys, sysconfig

def observation():
    prefix = pathlib.Path(sys.prefix)
    assert sys.flags.isolated == 1 and sys.dont_write_bytecode
    assert pathlib.Path(sys.executable).is_relative_to(prefix)
    assert all(pathlib.Path(p).is_relative_to(prefix) for p in sys.path)
    assert sys.prefix == sys.base_prefix
    assert hashlib.sha256(b"relocation").hexdigest()
    assert sqlite3.connect(":memory:").execute("select 1").fetchone() == (1,)
    return {"pid":os.getpid(),"prefix":sys.prefix,"executable":sys.executable,
            "isolated":sys.flags.isolated,"dont_write_bytecode":sys.dont_write_bytecode,
            "sys_path":sys.path,"ssl":ssl.OPENSSL_VERSION,
            "include_path":sysconfig.get_path("include"),"configured_libdir":sysconfig.get_config_var("LIBDIR")}

def child(queue):
    queue.put(observation())

if __name__ == "__main__":
    parent = observation()
    ctx=multiprocessing.get_context("spawn")
    queue=ctx.Queue()
    process=ctx.Process(target=child,args=(queue,))
    process.start()
    result=queue.get(timeout=30)
    process.join(timeout=30)
    assert process.exitcode == 0
    assert result["prefix"] == parent["prefix"]
    assert result["pid"] != parent["pid"]
    print(json.dumps({"qualification":"interpreter_relocation_only","parent":parent,"child":result,"model_loaded":False,"gpu_used":False}))
