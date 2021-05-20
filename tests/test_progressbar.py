import pytest
from tlo import progressbar


def test_in_zmq_interactive_shell():
    ret_val = progressbar._in_zmq_interactive_shell()
    # Assume tests never run from interactive shell (e.g. in Jupyter notebook)
    assert isinstance(ret_val, bool) and not ret_val


def test_in_shell_with_ansi_support():
    ret_val = progressbar._in_shell_with_ansi_support()
    # As tests may be run in shells with or without ANSI support only check if bool
    assert isinstance(ret_val, bool)


def test_create_display():
    ret_val = progressbar._create_display("")
    # Assume tests never run from interactive shell (e.g. in Jupyter notebook) so
    # that a StreamDisplay instance is always returned
    assert isinstance(ret_val, progressbar.StreamDisplay)


def test_format_stat():
    assert progressbar._format_stat(100) == "100"
    assert progressbar._format_stat(0.12341) == "0.1234"
    assert progressbar._format_stat("abc") == "abc"


def test_format_time():
    assert progressbar._format_time(100) == "01:40"
    assert progressbar._format_time(3661) == "1:01:01"


def test_ansi_stream_display(capfd):
    display = progressbar.AnsiStreamDisplay()
    display.update("Test")
    captured = capfd.readouterr()
    assert "Test" in captured.out


def test_basic_stream_display(capfd):
    display = progressbar.BasicStreamDisplay()
    display.update("Test")
    captured = capfd.readouterr()
    assert "Test" in captured.out


@pytest.mark.parametrize("n_step", ((1, 5, 10)))
def test_progress_bar_init(n_step):
    description = "Test"
    n_col = 10
    pb = progressbar.ProgressBar(
        n_step=n_step, description=description, n_col=n_col
    )
    assert pb.n_step == n_step
    assert pb.description == description
    assert pb.step == 0
    assert pb.prop_complete == 0
    assert pb.perc_complete == "  0%"
    assert pb.elapsed_time == "00:00"
    assert pb.iter_rate == "?"
    assert pb.est_remaining_time == "?"
    assert pb.n_block_filled == 0
    assert pb.n_block_empty == n_col
    assert pb.prop_partial_block == 0
    assert pb.filled_blocks == ""
    assert len(pb.empty_blocks) == n_col
    assert len(pb.progress_bar) == n_col + 2
    assert isinstance(pb.bar_color, str)
    assert pb.stats == ""
    assert isinstance(pb.prefix, str)
    assert isinstance(pb.postfix, str)
    assert isinstance(str(pb), str)


def test_progress_bar_raises():
    with pytest.raises(AssertionError, match="positive integer"):
        progressbar.ProgressBar(n_step=0)
    with pytest.raises(AssertionError, match="positive integer"):
        progressbar.ProgressBar(n_step=-1)
    with pytest.raises(AssertionError, match="positive integer"):
        progressbar.ProgressBar(n_step=1, n_col=0)
    with pytest.raises(AssertionError, match="positive integer"):
        progressbar.ProgressBar(n_step=1, n_col=-1)
    with pytest.raises(AssertionError, match="non-negative"):
        progressbar.ProgressBar(n_step=1, n_col=1, min_refresh_time=-1)


@pytest.mark.parametrize("n_step", ((1, 5, 10)))
def test_progress_bar_update(capfd, n_step):
    description = "Test"
    pb = progressbar.ProgressBar(
        n_step=n_step, description=description, min_refresh_time=0
    )
    pb.start()
    pb_strings = [str(pb)]
    for i in range(n_step):
        pb.update(step=i + 1, stats_dict={"stat": i})
        assert pb.elapsed_time != "?"
        assert 0 < pb.prop_complete <= 1
        pb_strings.append(str(pb))
    pb.stop()
    captured_strings = capfd.readouterr().out.split("\r")[1:]
    for string_set in (pb_strings, captured_strings):
        for i, string in enumerate(string_set):
            assert description in string
            assert f"{i}/{n_step}" in string
            if i > 0:
                assert f"stat={i - 1}" in string
