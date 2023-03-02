# pylint: disable=protected-access, attribute-defined-outside-init, missing-function-docstring

""" Test Callback abstract classes """

import pytest
import pytest_check as check
from benchmarks.callbacks.callback import Callback, CallbackList


class TestCallback:

    """Callback"""

    def test_init(self):
        """should instanciate correctly."""
        callback = Callback()
        check.equal(callback.params, {})
        check.is_none(callback.playground)

    def test_set_params(self):
        """should update params correctly."""
        callback = Callback()
        expected_params = {"param": 1}
        callback.set_params(expected_params)
        check.equal(callback.params, expected_params)

    def test_set_playground(self):
        """should update playground correctly."""
        callback = Callback()
        expected_playground = "playground"
        callback.set_playground(expected_playground)
        check.equal(callback.playground, expected_playground)

    @pytest.mark.parametrize("hook", ["begin", "end"])
    @pytest.mark.parametrize(
        "timescale", ["step", "steps_cycle", "episode", "episodes_cycle", "run"]
    )
    def test_callback_on_(self, timescale, hook):
        callback = Callback()
        callback_hook = getattr(callback, f"on_{timescale}_{hook}")
        if timescale in ["step", "steps_cycle"]:
            callback_hook(step=7, logs={"log": "info"})
        elif timescale in ["episode", "episodes_cycle"]:
            callback_hook(episode=7, logs={"log": "info"})
        else:
            callback_hook(logs={"log": "info"})


class TestCallbackList:

    """CallbackList"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.callback1 = Callback()
        self.callback2 = Callback()
        self.callbacks = [self.callback1, self.callback2]
        self.callback_path = "benchmarks.callbacks.callback.Callback"

    def test_init(self):
        """should instanciate correctly."""
        callbacks = CallbackList(self.callbacks)
        check.equal(callbacks.params, {})
        check.is_none(callbacks.playground)

    def test_set_params(self, mocker):
        """should update params of all callbacks correctly."""
        mocker.patch(self.callback_path + ".set_params")

        callbacks = CallbackList(self.callbacks)
        expected_params = {"param": 1}
        callbacks.set_params(expected_params)

        check.equal(callbacks.params, expected_params)
        for callback in self.callbacks:
            args, _ = callback.set_params.call_args
            check.equal(args[0], expected_params)

    def test_set_playground(self, mocker):
        """should update playground of all callbacks correctly."""
        mocker.patch(self.callback_path + ".set_playground")

        callbacks = CallbackList(self.callbacks)
        expected_playground = "playground"
        callbacks.set_playground(expected_playground)

        check.equal(callbacks.playground, expected_playground)
        for callback in self.callbacks:
            args, _ = callback.set_playground.call_args
            check.equal(args[0], expected_playground)

    def test_call_key_hook_no_callback(self):
        """_call_key_hook should do nothing if no callbacks."""
        callbacks = CallbackList([])
        callbacks._call_key_hook("timescale", "at")

    @pytest.mark.parametrize("hook", ["begin", "end"])
    @pytest.mark.parametrize(
        "timescale", ["step", "steps_cycle", "episode", "episodes_cycle", "run"]
    )
    def test_call_key_hook(self, mocker, timescale, hook):
        time = 0.1 if hook == "begin" else 0.123
        value = 17 if timescale != "run" else None
        logs = {"log": 7}
        t_begin_name = f"t_{timescale}_begin"
        dt_name = f"dt_{timescale}"
        hook_name = f"on_{timescale}_{hook}"

        mocker.patch("time.time", return_value=time)
        mocker.patch("benchmarks.callbacks.callback.Callback." + hook_name)

        callbacks = CallbackList(self.callbacks)
        if hook == "end":
            setattr(callbacks, t_begin_name, 0.1)

        callbacks._call_key_hook(timescale, hook, value=value, logs=logs)

        if hook == "end":
            check.almost_equal(logs.get(dt_name), 0.023)

        for callback in self.callbacks:
            check.is_true(getattr(callback, hook_name).called)

    @pytest.mark.parametrize("hook", ["begin", "end"])
    @pytest.mark.parametrize(
        "timescale", ["step", "steps_cycle", "episode", "episodes_cycle", "run"]
    )
    def test_calls_all_callbacks_on_(self, timescale, hook, mocker):
        mocker.patch("benchmarks.callbacks.callback.CallbackList._call_key_hook")

        callbacks = CallbackList(self.callbacks)
        callback_hook = getattr(callbacks, f"on_{timescale}_{hook}")
        if timescale in ["step", "steps_cycle"]:
            callback_hook(step=7, logs={"log": "info"})
        elif timescale in ["episode", "episodes_cycle"]:
            callback_hook(episode=7, logs={"log": "info"})
        else:
            callback_hook(logs={"log": "info"})

        args, _ = callbacks._call_key_hook.call_args
        check.equal(args[0], timescale)
        check.equal(args[1], hook)
