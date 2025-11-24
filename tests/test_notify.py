from tlo.notify import notifier

def test_notifier():
    # in listening code
    received_data = []

    def callback(data):
        received_data.append(data)

    notifier.add_listener("test.signal", callback)

    # in emitting code
    notifier.dispatch("test.signal", data={"value": 42})

    assert len(received_data) == 1
    assert received_data[0] == {"value": 42}

    # Unsubscribe and test no further calls
    notifier.remove_listener("test.signal", callback)
    notifier.dispatch("test.signal", data={"value": 100})

    assert len(received_data) == 1  # No new data

