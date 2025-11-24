"""
A dead simple synchronous notification dispatcher.

Usage
-----
# In the notifying class/module
from tlo.notify import notifier

notifier.dispatch("simulation.on_start", data={"one": 1, "two": 2})

# In the listening class/module
from tlo.notify import notifier

def on_notification(data):
    print("Received notification:", data)

notifier.add_listener("simulation.on_start", on_notification)
"""


class Notifier:
    """
    A simple synchronous notification dispatcher supporting listeners.
    """

    def __init__(self):
        self.listeners = {}

    def add_listener(self, notification_key, listener):
        """
        Register a listener for a specific notification.

        :param notification_key: The identifier to listen for.
        :param listener: A callable to be invoked when the notification is dispatched.
        """
        if notification_key not in self.listeners:
            self.listeners[notification_key] = []
        self.listeners[notification_key].append(listener)

    def remove_listener(self, notification_key, listener):
        """
        Remove a previously registered listener for a notification.

        :param notification_key: The identifier.
        :param listener: The listener callable to remove.
        """
        if notification_key in self.listeners:
            self.listeners[notification_key].remove(listener)
            if not self.listeners[notification_key]:
                del self.listeners[notification_key]

    def dispatch(self, notification_key, data=None):
        """
        Dispatch a notification to all registered listeners.

        :param notification_key: The identifier.
        :param data: Optional data to pass to each listener.
        """
        if notification_key in self.listeners:
            for listener in list(self.listeners[notification_key]):
                listener(data)


# Create a global notifier instance
notifier = Notifier()
