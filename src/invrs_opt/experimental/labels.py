"""Defines labels and messages used in the context of an optimization service.

Copyright (c) 2023 The INVRS-IO authors.
"""

VALUE = "value"
GRAD = "grad"
PARAMS = "params"
STATE_TOKEN = "state_token"
MESSAGE = "message"

OPT_CONFIG = "opt_config"
ALGORITHM = "algorithm"
HPARAMS = "hparams"
DATA = "data"

ROUTE_INIT = "init"
ROUTE_UPDATE = "update"
ROUTE_PARAMS = "params"

MESSAGE_STATE_NOT_KNOWN = "State token {} was not recognized."
MESSAGE_STATE_NOT_READY = "State for token {} is not ready."
MESSAGE_STATE_NOT_VALID = "State for token {} is not valid."
