class CloudpickleWrapper:
    """Wrapper to enable multiprocessing with cloudpickle serialization."""

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)
