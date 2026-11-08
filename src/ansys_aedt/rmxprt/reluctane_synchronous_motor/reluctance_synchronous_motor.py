from .configuation import Configuation
from .utils.create_model import create_model

class ReluctanceSynchronousMotor:
    def __init__(self):
        self.configuation = Configuation()
        self.rmxprt = None
        self.m2d = None

    def create_model(self):
        return create_model(reluctance_synchronous_motor=self)