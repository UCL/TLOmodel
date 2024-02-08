"""Progress bar for visualising progress of simulation runs."""

import html
import os
import platform
import sys
from timeit import default_timer as timer
from typing import Dict, Optional, TextIO, Union

try:
    from IPython import get_ipython
    from IPython.display import display as ipython_display

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


def _in_zmq_interactive_shell() -> bool:
    """Check if in interactive ZMQ shell which supports updateable displays"""
    if not IPYTHON_AVAILABLE:
        return False
    else:
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return True
            elif shell == "TerminalInteractiveShell":
                return False
            else:
                return False
        except NameError:
            return False


def _in_shell_with_ansi_support() -> bool:
    """Check if running in shell with support for ANSI escape characters.

    Based on https://gist.github.com/ssbarnea/1316877
    """
    return (
        (
            hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
            and platform.system() != "Windows"
        )
        or os.environ.get("TERM") == "ANSI"
        or os.environ.get("PYCHARM_HOSTED") == "1"
    )


def _create_display(obj):
    """Create an updateable display object.

    :param obj: Initial object to display.
    :return: Object with `update` method to update displayed content.
    """
    if _in_zmq_interactive_shell():
        return ipython_display(obj, display_id=True)
    else:
        display = (
            AnsiStreamDisplay()
            if _in_shell_with_ansi_support()
            else BasicStreamDisplay()
        )
        display.update(obj)
        return display


def _format_time(total_seconds: Union[int, float]) -> str:
    """Format a time interval in seconds as a colon-delimited string [h:]m:s"""
    total_mins, seconds = divmod(int(total_seconds), 60)
    hours, mins = divmod(total_mins, 60)
    if hours != 0:
        return f"{hours:d}:{mins:02d}:{seconds:02d}"
    else:
        return f"{mins:02d}:{seconds:02d}"


def _format_stat(stat) -> str:
    """Format numeric stat at fixed precision otherwise convert to string."""
    if isinstance(stat, (int, float)):
        return f"{stat:.4g}"
    else:
        return str(stat)


class ProgressBar:
    """Iterable object for tracking progress of an iterative task.

    Implements both string and HTML representations to allow richer
    display in interfaces which support HTML output, for example Jupyter
    notebooks or interactive terminals.
    """

    GLYPHS = " ▏▎▍▌▋▊▉█"
    """Characters used to create string representation of progress bar."""

    def __init__(
        self,
        n_step: int,
        description: Optional[str] = None,
        n_col: int = 10,
        unit: str = "step",
        min_refresh_time: float = 1.0,
    ):
        """
        :param n_step: Total number of steps in task.
        :param description: Description of task to prefix progress bar with.
        :param n_col: Number of columns (characters) to use in string
            representation of progress bar.
        :param unit: String describing unit of each step.
        :param min_referesh_time: Minimum time in seconds between each
            refresh of progress bar visual representation.
        """
        assert int(n_step) == n_step and n_step > 0, "n_step must be positive integer"
        self._n_step = int(n_step)
        self._description = description
        self._active = False
        assert int(n_col) == n_col and n_col > 0, "n_col must be positive integer"
        self._n_col = int(n_col)
        self._unit = unit
        self._capitalized_unit = unit.capitalize()
        self._step = 0
        self._start_time = None
        self._elapsed_time = 0
        self._stats_dict = {}
        assert min_refresh_time >= 0, "min_refresh_time must be non-negative"
        self._min_refresh_time = min_refresh_time
        self._display = None

    @property
    def n_step(self):
        """Total number of steps in task."""
        return self._n_step

    @property
    def description(self):
        """ "Description of task being tracked."""
        return self._description

    @property
    def step(self):
        """Progress step count."""
        return self._step

    @step.setter
    def step(self, value):
        self._step = max(0, min(value, self.n_step))

    @property
    def prop_complete(self):
        """Proportion complete (float value in [0, 1])."""
        return self.step / self.n_step

    @property
    def perc_complete(self):
        """Percentage complete formatted as string."""
        return f"{int(self.prop_complete * 100):3d}%"

    @property
    def elapsed_time(self):
        """Elapsed time formatted as string."""
        return _format_time(self._elapsed_time)

    @property
    def iter_rate(self):
        """Mean iteration rate if ≥ 1 `unit/s` or reciprocal `s/unit` as string."""
        if self.prop_complete == 0:
            return "?"
        else:
            mean_time = self._elapsed_time / self.step
            return (
                f"{mean_time:.2f}s/{self._unit}"
                if mean_time > 1
                else f"{1/mean_time:.2f}{self._unit}/s"
            )

    @property
    def est_remaining_time(self):
        """Estimated remaining time to completion formatted as string."""
        if self.prop_complete == 0:
            return "?"
        else:
            return _format_time((1 / self.prop_complete - 1) * self._elapsed_time)

    @property
    def n_block_filled(self):
        """Number of filled blocks in progress bar."""
        return int(self._n_col * self.prop_complete)

    @property
    def n_block_empty(self):
        """Number of empty blocks in progress bar."""
        return self._n_col - self.n_block_filled

    @property
    def prop_partial_block(self):
        """Proportion filled in partial block in progress bar."""
        return self._n_col * self.prop_complete - self.n_block_filled

    @property
    def filled_blocks(self):
        """Filled blocks string."""
        return self.GLYPHS[-1] * self.n_block_filled

    @property
    def empty_blocks(self):
        """Empty blocks string."""
        if self.prop_partial_block == 0:
            return self.GLYPHS[0] * self.n_block_empty
        else:
            return self.GLYPHS[0] * (self.n_block_empty - 1)

    @property
    def partial_block(self):
        """Partial block character."""
        if self.prop_partial_block == 0:
            return ""
        else:
            return self.GLYPHS[int(len(self.GLYPHS) * self.prop_partial_block)]

    @property
    def progress_bar(self):
        """Progress bar string."""
        return f"|{self.filled_blocks}{self.partial_block}{self.empty_blocks}|"

    @property
    def bar_color(self):
        """CSS color property for HTML progress bar."""
        if self.step == self.n_step:
            return "var(--jp-success-color1, #4caf50)"
        elif self._active:
            return "var(--jp-brand-color1, #2196f3)"
        else:
            return "var(--jp-error-color1, #f44336)"

    @property
    def stats(self):
        """Comma-delimited string list of statistic key=value pairs."""
        return ", ".join(f"{k}={_format_stat(v)}" for k, v in self._stats_dict.items())

    @property
    def prefix(self):
        """Text to prefix progress bar with."""
        return (
            f'{self.description + ": "if self.description else ""}'
            f"{self.perc_complete}"
        )

    @property
    def postfix(self):
        """Text to postfix progress bar with."""
        return (
            f"{self._capitalized_unit} {self.step}/{self.n_step} "
            f"[{self.elapsed_time}<{self.est_remaining_time}, "
            f"{self.iter_rate}"
            f'{", " + self.stats if self._stats_dict else ""}]'
        )

    def reset(self):
        """Reset progress bar state."""
        self._step = 0
        self._start_time = timer()
        self._last_refresh_time = -float("inf")
        self._stats_dict = {}

    def update(
        self, step: int, stats_dict: Optional[Dict] = None, refresh: bool = True
    ):
        """Update progress bar state.

        :param step: New value for step counter.
        :param stats_dict: Dictionary of statistic key-value pairs to use to
            update postfix stats.
        :param refresh: Whether to refresh display.
        """
        if step == 0:
            self.reset()
        else:
            self.step = step
            if stats_dict is not None:
                self._stats_dict.update(stats_dict)
            self._elapsed_time = timer() - self._start_time
        if (
            refresh
            and step == self.n_step
            or (timer() - self._last_refresh_time > self._min_refresh_time)
        ):
            self.refresh()
            self._last_refresh_time = timer()

    def refresh(self):
        """Refresh visual display(s) of progress bar."""
        self._display.update(self)

    def start(self):
        """Start tracking progress of task."""
        self._active = True
        self.reset()
        if self._display is None:
            self._display = _create_display(self)

    def stop(self):
        """Stop tracking progress of task."""
        self._active = False
        if self.step != self.n_step:
            self.refresh()
        if isinstance(self._display, StreamDisplay):
            self._display.close()

    def __str__(self):
        return f"{self.prefix}{self.progress_bar}{self.postfix}"

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        return f"""
        <div style="line-height: 28px; width: 100%; display: flex;
                    flex-flow: row wrap; align-items: center;
                    position: relative; margin: 2px;">
          <label style="margin-right: 8px; flex-shrink: 0;
                        font-size: var(--jp-code-font-size, 13px);
                        font-family: var(--jp-code-font-family, monospace);">
            {html.escape(self.prefix).replace(' ', '&nbsp;')}
          </label>
          <div role="progressbar" aria-valuenow="{self.prop_complete}"
               aria-valuemin="0" aria-valuemax="1"
               style="position: relative; flex-grow: 1; align-self: stretch;
                      margin-top: 4px; margin-bottom: 4px;  height: initial;
                      background-color: #eee;">
            <div style="background-color: {self.bar_color}; position: absolute;
                        bottom: 0; left: 0; width: {self.perc_complete};
                        height: 100%;"></div>
          </div>
          <div style="margin-left: 8px; flex-shrink: 0;
                      font-family: var(--jp-code-font-family, monospace);
                      font-size: var(--jp-code-font-size, 13px);">
            {html.escape(self.postfix)}
          </div>
        </div>
        """


class StreamDisplay:
    """Base class for using I/O streams as an updatable display."""

    def __init__(self, io: Optional[TextIO] = None):
        """
        :param io: I/O stream to write updates to. Defaults to `sys.stdout` if `None`.
        """
        self._io = io if io is not None else sys.stdout

    def close(self):
        self._io.write("\n")
        self._io.flush()

    def update(self, obj):
        """Update display with string representation of `obj`."""
        raise NotImplementedError()


class AnsiStreamDisplay(StreamDisplay):
    """Use I/O stream which supports ANSI escape sequences as an updatable display."""

    def update(self, obj):
        self._io.write("\x1b[2K\r")
        self._io.write(str(obj))
        self._io.flush()


class BasicStreamDisplay(StreamDisplay):
    """Use I/O stream without ANSI escape sequence support as an updatable display."""

    def __init__(self, io: Optional[TextIO] = None):
        super().__init__(io)
        self._last_string_length = 0

    def update(self, obj):
        string = str(obj)
        self._io.write(f"\r{string: <{self._last_string_length}}")
        self._last_string_length = len(string)
        self._io.flush()
