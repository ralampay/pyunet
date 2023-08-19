class ExtractParamsFromConfig:
    def __init__(self, config_file):
        self.config_file    = config_file

        self.params = {
        }

    def execute(self):
        print(f"Config file {self.config_file}")
