from enum import Enum

class Task(Enum):
    GRAPH_CLASSIFICATION = 1
    NODE_CLASSIFICATION = 2
    LINK_PREDICTION = 3

class Stage(Enum):
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3

class Experiment(Enum):
    DEFAULT = 1
    GREEDY = 2
    NO_Q = 3
    RANDOM = 4