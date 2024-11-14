def worker(remote, parent_remote, env_fn_wrapper):
    """Worker process to handle environment interactions asynchronously."""
    parent_remote.close()
    env = env_fn_wrapper.x
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                ob, reward, done, info, _ = env.step(data)
                if done:
                    ob = env.reset()[0]
                remote.send((ob, reward, done, info))
            elif cmd == "reset":
                ob = env.reset()[0]
                remote.send(ob)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.get_observation_space(), env.get_action_space()))
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
        except EOFError:
            print("EOFError: Worker encountered an issue and will shut down.")
            break
