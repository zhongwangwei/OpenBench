

class statistics_calculate:
    """
    A class for performing various statistical analyses on xarray datasets.
    """
    def __init__(self, info):
        """
        Initialize the Statistics class.

        Args:
            info (dict): A dictionary containing additional information to be added as attributes.
        """
        self.name = 'statistics'
        self.version = '0.2'
        self.release = '0.2'
        self.date = 'Mar 2024'
        self.author = "Zhongwang Wei"
        self.__dict__.update(info)