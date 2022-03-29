from ignite.engine import Events
from ignite.handlers import Timer

__all__ = ["ModelTimer"]


class ModelTimer(object):
    """Timer object can be used to measure average time.

    This timer class compute averaged time for:
        - "batch": time of each iteration
        - "data": time of getting batch data
        - "forward": time of model forwarding

    Example:

        >>> from ignite.engine import Engine, Events
        >>> from ignite.handlers import Timer
        >>> trainer = Engine(function)
        >>> timer = ModelTimer()
        >>> timer.attach(trainer)

    Version:
        - ignite>=0.4.2
    """

    def __init__(self):
        self.data_timer = Timer(average=True)
        self.nn_timer = Timer(average=True)
        self.timer = Timer(average=True)

    def attach(self, enginer):
        self.data_timer.attach(
            enginer,
            start=Events.EPOCH_STARTED,
            resume=Events.GET_BATCH_STARTED,
            pause=Events.GET_BATCH_COMPLETED,
            step=Events.ITERATION_COMPLETED,
        )
        self.nn_timer.attach(
            enginer,
            start=Events.EPOCH_STARTED,
            resume=Events.GET_BATCH_COMPLETED,
            pause=Events.GET_BATCH_STARTED,
            step=Events.ITERATION_COMPLETED,
        )
        self.timer.attach(
            enginer, start=Events.EPOCH_STARTED, step=Events.ITERATION_COMPLETED,
        )

    def __str__(self):
        return "{:.3f} batch, {:.3f} data, {:.3f} forward".format(
            self.timer.value(), self.data_timer.value(), self.nn_timer.value()
        )

    def value(self):
        """Return averaged time for each phase.

        Returns:
            dict: time
        """
        time = {"data": self.data_timer.value(), "batch": self.timer.value(), "forward": self.nn_timer.value()}
        return time
