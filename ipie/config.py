class Config:

    def __init__(self):
        self.options = {}

    def add_option(self, key, val):
        self.options[key] = val

    def update(self, key, val):
        _val = self.options.get(key)
        if _val is None:
            raise KeyError("config option not found: {}".format(_val))
        self.options[key] = val

    def get_option(self, key):
        _val = self.options.get(key)
        if _val is None:
            raise KeyError("config option not found: {}".format(_val))
        return _val

    def __str__(self):
        _str = ''
        for k, v in self.options.items():
            _str += '{} : {}\n'.format(k, v)
        return _str

config = Config()

# Default to not using for the moment.
config.add_option('use_gpu', False)
# Memory limits should be in GB
config.add_option('max_memory_for_wicks', 2.0)
config.add_option('max_memory_sd_energy_gpu', 2.0)
