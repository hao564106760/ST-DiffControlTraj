import logging
import sys
from copy import copy
from typing import Union

from colored import attr, fg

DEBUG = "debug"
INFO = "info"
WARNING = "warning"
ERROR = "error"
CRITICAL = "critical"

LOG_LEVELS = {
    DEBUG: logging.DEBUG,
    INFO: logging.INFO,
    WARNING: logging.WARNING,
    ERROR: logging.ERROR,
    CRITICAL: logging.CRITICAL,
}


class _Formatter(logging.Formatter):
    def __init__(self, colorize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colorize = colorize

    @staticmethod
    def _paint(msg, level, colorize):
        lvl = str(level).lower()
        if lvl not in LOG_LEVELS:
            raise RuntimeError(f"bad log level: {lvl}")

        s = f"{lvl.upper()}: {msg}"
        if not colorize:
            return s

        if lvl == DEBUG:
            return f"{fg(5)}{s}{attr(0)}"
        if lvl == INFO:
            return f"{fg(4)}{s}{attr(0)}"
        if lvl == WARNING:
            return f"{fg(214)}{attr(1)}{s}{attr(21)}{attr(0)}"
        if lvl == ERROR:
            return f"{fg(202)}{attr(1)}{s}{attr(21)}{attr(0)}"
        return f"{fg(196)}{attr(1)}{s}{attr(21)}{attr(0)}"

    def format(self, record):
        r = copy(record)
        r.msg = self._paint(r.msg, r.levelname, self.colorize)
        return super().format(r)


class Logger:
    def __init__(
        self,
        name="default",
        colorize=False,
        log_path=None,
        stream=sys.stdout,
        level=INFO,
    ):
        self.name = name
        self.__logger = logging.getLogger(f"_logger-{name}")
        self.__logger.propagate = False
        self.clear_handlers()

        self.__formatter = _Formatter(
            colorize=colorize,
            fmt="[%(process)d][%(asctime)s.%(msecs)03d @ %(funcName)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
        )

        self.setLevel(level)

        self.__stream_to_handler = {}
        self.__main_handler = self.add_handler(stream)

        if log_path is not None:
            fh = logging.FileHandler(log_path, mode="w")
            fh.setFormatter(self.__formatter)
            self.__logger.addHandler(fh)

        self.debug = self.__logger.debug
        self.info = self.__logger.info
        self.warning = self.__logger.warning
        self.error = self.__logger.error
        self.critical = self.__logger.critical

    def setLevel(self, level: Union[str, int]) -> None:
        if isinstance(level, int):
            self.__logger.setLevel(level)
            return
        lvl = str(level).lower()
        if lvl not in LOG_LEVELS:
            raise ValueError(f"bad level: {lvl}")
        self.__logger.setLevel(LOG_LEVELS[lvl])

    def add_handler(self, stream) -> logging.StreamHandler:
        h = logging.StreamHandler(stream)
        h.setFormatter(self.__formatter)
        self.__logger.addHandler(h)
        self.__stream_to_handler[stream] = h
        return h

    def remove_handler(self, stream) -> bool:
        h = self.__stream_to_handler.get(stream)
        if h is None:
            return False
        self.__logger.removeHandler(h)
        self.__stream_to_handler.pop(stream, None)
        return True

    def clear_handlers(self) -> None:
        self.__logger.handlers = []
        self.__stream_to_handler = {}

    def get_handlers(self) -> list:
        return self.__logger.handlers

    @property
    def inner_logger(self):
        return self.__logger

    @property
    def inner_stream_handler(self):
        return self.__main_handler

    @property
    def inner_formatter(self):
        return self.__formatter


def log_info(args, logger):
    # Minimal run config dump
    logger.info("==== config ====")
    logger.info(f"dataset={args.data.dataset}")
    logger.info(f"traj_len={args.data.traj_length}")
    logger.info(f"guidance={args.model.guidance_scale}")
    logger.info(f"steps={args.diffusion.num_diffusion_timesteps}")
    logger.info(f"beta_sched={args.diffusion.beta_schedule}")
    logger.info(f"beta_start={args.diffusion.beta_start}")
    logger.info(f"beta_end={args.diffusion.beta_end}")
    logger.info(f"epochs={args.training.n_epochs}")
    logger.info(f"batch={args.training.batch_size}")
    logger.info("===============")
