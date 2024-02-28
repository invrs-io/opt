"""Defines basic client optimizers for use with an optimization service.

Copyright (c) 2023 The INVRS-IO authors.
"""

import json
import requests
import time
from typing import Any, Dict, Optional

from totypes import json_utils

from invrs_opt import base
from invrs_opt.experimental import labels


PyTree = Any
StateToken = str

SESSION = None
SERVER_ADDRESS = None


def login(server_address: str) -> None:
    """Set the global server address and create a requests session."""
    global SESSION
    global SERVER_ADDRESS
    SESSION = requests.Session()
    SERVER_ADDRESS = server_address


def optimizer_client(
    algorithm: str,
    hparams: Dict[str, Any],
    server_address: Optional[str],
    session: Optional[requests.Session],
) -> base.Optimizer:
    """Generic optimizer class."""

    if server_address is None:
        if SERVER_ADDRESS is None:
            raise ValueError(
                "Argument `server_address` and the global `SERVER_ADDRESS` cannot "
                "both be `None`. Use the `login` method to set the global, or "
                "explicitly provide a value."
            )
    if session is None:
        if SESSION is None:
            raise ValueError(
                "Argument `session` and the global `SESSION` cannot "
                "both be `None`. Use the `login` method to set the global, or "
                "explicitly provide a value."
            )
        session = SESSION

    opt_config = {
        labels.ALGORITHM: algorithm,
        labels.HPARAMS: hparams,
    }

    def init_fn(params: PyTree) -> StateToken:
        """Handles 'init' requests."""
        serialized_data = json_utils.json_from_pytree(
            dict(opt_config=opt_config, data={"params": params})
        )
        post_response = session.post(
            f"{SERVER_ADDRESS}/{labels.ROUTE_INIT}/", data=serialized_data
        )

        if not post_response.status_code == 200:
            raise requests.RequestException(post_response.text)
        response = json.loads(post_response.text)
        new_state_token: str = response[labels.STATE_TOKEN]
        return new_state_token

    def update_fn(
        *,
        grad: PyTree,
        value: float,
        params: PyTree,
        state: StateToken,
    ) -> StateToken:
        """Handles 'update' requests."""
        state_token = state
        del state
        serialized_data = json_utils.json_from_pytree(
            {
                labels.OPT_CONFIG: opt_config,
                labels.DATA: {
                    labels.PARAMS: params,
                    labels.VALUE: value,
                    labels.GRAD: grad,
                    labels.STATE_TOKEN: state_token,
                },
            }
        )
        post_response = session.post(
            f"{SERVER_ADDRESS}/{labels.ROUTE_UPDATE}/{state_token}/",
            data=serialized_data,
        )

        if not post_response.status_code == 200:
            raise requests.RequestException(post_response.text)
        response = json.loads(post_response.text)
        new_state_token: str = response[labels.STATE_TOKEN]
        return new_state_token

    def params_fn(
        state: StateToken,
        timeout: float = 60.0,
        poll_interval: float = 0.1,
    ) -> PyTree:
        """Handles 'params' requests."""
        state_token = state
        del state
        assert timeout >= poll_interval
        start_time = time.time()
        while time.time() < start_time + timeout:
            get_response = session.get(
                f"{SERVER_ADDRESS}/{labels.ROUTE_PARAMS}/{state_token}"
            )
            if get_response.status_code == 200:
                break
            elif get_response.status_code == 404 and get_response.text.endswith(
                labels.MESSAGE_STATE_NOT_READY.format(state_token)
            ):
                time.sleep(poll_interval)
            else:
                raise requests.RequestException(get_response.text)

        if not get_response.status_code == 200:
            raise requests.Timeout("Timed out while waiting for params.")
        response = json.loads(get_response.text)
        return json_utils.pytree_from_json(response[labels.PARAMS])

    return base.Optimizer(init=init_fn, update=update_fn, params=params_fn)


# -----------------------------------------------------------------------------
# Specific optimizers implemented here.
# -----------------------------------------------------------------------------


def lbfgsb(maxcor: int = 20, line_search_max_steps: int = 100) -> base.Optimizer:
    """Optimizer implementing the L-BFGS-B scheme."""
    hparams = {
        "maxcor": maxcor,
        "line_search_max_steps": line_search_max_steps,
    }
    return optimizer_client(
        algorithm="lbfgsb", hparams=hparams, server_address=None, session=None
    )
