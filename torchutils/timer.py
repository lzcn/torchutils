from time import perf_counter
from typing import Any

from ignite.engine import EventEnum


class States(EventEnum):
    INITED = "inited"

    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"

    STARTED = "started"
    COMPLETED = "completed"

    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"

    MODEL_STARTED = "model_started"
    MODEL_COMPLETED = "model_completed"

    DATA_STARTED = "get_batch_started"
    DATA_COMPLETED = "get_batch_completed"


class Timer(object):
    """Timer object. Also see :class:`~ignite.handlers.Timer`.

    Args:
        average: if True, then when ``.value()`` method is called, the returned value
            will be equal to total time measured, divided by the value of internal counter.

    Attributes:
        total (float): total time elapsed when the Timer was running (in seconds).
        step_count (int): internal counter, useful to measure average time, e.g. of processing a single batch.
            Incremented with the ``.step()`` method.
        running (bool): flag indicating if timer is measuring time.

    Note:
        When using ``Timer(average=True)`` do not forget to call ``timer.step()`` every time an event occurs. See
        the examples below.

    Examples:
        Measuring total time of the epoch:

        .. code-block:: python

            from torchutils.timer import Timer
            import time
            work = lambda : time.sleep(0.1)
            idle = lambda : time.sleep(0.1)
            t = Timer(average=False)
            for _ in range(10):
                work()
                idle()

            t.value()
            # 2.003073937026784

        Measuring average time of the epoch:

        .. code-block:: python

            t = Timer(average=True)
            for _ in range(10):
                work()
                idle()
                t.step()

            t.value()
            # 0.2003182829997968

        Measuring average time it takes to execute a single ``work()`` call:

        .. code-block:: python

            t = Timer(average=True)
            for _ in range(10):
                t.resume()
                work()
                t.pause()
                idle()
                t.step()

            t.value()
            # 0.10016545779653825

    """

    def __init__(self, average: bool = False):
        self._average = average

        self.reset()

    def reset(self, *args: Any) -> "Timer":
        """Reset the timer to zero."""
        self._t0 = perf_counter()
        self.total = 0.0
        self.step_count = 0.0
        self.running = True

        return self

    def pause(self, *args: Any) -> None:
        """Pause the current running timer.

        Pausing a timer will add the elapsed time from last successful resuming and makes
        the time inactive.
        """
        if self.running:
            self.total += self._elapsed()
            self.running = False

    def resume(self, *args: Any) -> None:
        """Resume the current running timer.

        Resuming a timer will change an inactive timer to running and record a t0.
        If the timer is already running, nothing will be changed.


        To force a timer to forget the previous t0, set the ``running`` attribute to True
        before calling ``resume``.
        """
        if not self.running:
            self.running = True
            self._t0 = perf_counter()

    def value(self) -> float:
        """Return the average timer value."""
        total = self.total
        if self.running:
            total += self._elapsed()

        if self._average:
            denominator = max(self.step_count, 1.0)
        else:
            denominator = 1.0

        return total / denominator

    def step(self, *args: Any) -> None:
        """Increment the counter for computing averaged elapsed time."""
        self.step_count += 1.0

    def _elapsed(self) -> float:
        return perf_counter() - self._t0


class ModelTimer(object):
    """Timer object can be used to measure average time.

    This timer class compute averaged time for:
        - "batch": time of each iteration
        - "data": time of getting batch data
        - "forward": time of model forwarding

    Examples:
        Measuring each phase of an iteration

        .. code-block:: python

            from torchutils.timer import ModelTimer

            timer = ModelTimer()
            timer.start()
            for n in range(epochs):
                // start training
                timer.epoch_start()
                for batch in dataloader:
                    # record the time during data batch
                    timer.data_complete()
                    // ...
                    loss = net(batch)
                    timer.model_complete()
                    // ...
                    timer.iteration_complete()
                timer.epoch_complete()
                // evaluation
    """

    def __init__(self, average=True):
        self.data_timer = Timer(average=average)
        self.model_timer = Timer(average=average)
        self.batch_timer = Timer(average=average)
        self.state = States.INITED

    def __str__(self):
        return "{:.3f} batch, {:.3f} data, {:.3f} forward".format(
            self.batch_timer.value(), self.data_timer.value(), self.model_timer.value()
        )

    def reset(self):
        self.data_timer.reset()
        self.model_timer.reset()
        self.batch_timer.reset()
        self.state = States.INITED

    def start(self):
        # make all timers inactive
        self.data_timer.running = False
        self.model_timer.running = False
        self.batch_timer.running = False
        # start recording time for batch iteration and data loading
        self.batch_timer.resume()
        self.data_timer.resume()
        self.state = States.STARTED

    def data_start(self):
        """Called when an iteration is completed."""
        # forget previous t0 and renew t0
        self.data_timer.running = False
        self.data_timer.resume()
        self.state = States.DATA_STARTED

    def data_complete(self):
        self.data_timer.pause()
        self.data_timer.step()
        self.model_timer.resume()
        self.state = States.DATA_COMPLETED

    def model_start(self):
        self.model_timer.running = False
        self.model_timer.resume()
        self.state = States.MODEL_STARTED

    def model_complete(self):
        self.model_timer.pause()
        self.model_timer.step()
        self.data_timer.resume()
        self.state = States.MODEL_COMPLETED

    def iteration_start(self):
        self.batch_timer.running = False
        self.batch_timer.resume()
        self.state = States.ITERATION_STARTED

    def iteration_complete(self):
        self.batch_timer.pause()
        self.batch_timer.step()
        if self.state is States.MODEL_COMPLETED:
            self.data_timer.running = False
            self.data_timer.resume()
        self.data_timer.resume()
        self.batch_timer.resume()
        self.state = States.ITERATION_COMPLETED

    def epoch_start(self):
        # make all timers inactive
        self.data_timer.running = False
        self.model_timer.running = False
        self.batch_timer.running = False
        # start recording time for batch iteration and data loading
        self.batch_timer.resume()
        self.data_timer.resume()
        self.state = States.EPOCH_STARTED

    def epoch_complete(self):
        self.data_timer.pause()
        self.model_timer.pause()
        self.batch_timer.pause()
        self.state = States.EPOCH_COMPLETED

    def complete(self):
        self.data_timer.pause()
        self.model_timer.pause()
        self.batch_timer.pause()
        self.state = States.COMPLETED

    def value(self):
        """Return averaged time for each phase.

        Returns:
            dict: time
        """
        time = {"batch": self.batch_timer.value(), "data": self.data_timer.value(), "model": self.model_timer.value()}
        return time
