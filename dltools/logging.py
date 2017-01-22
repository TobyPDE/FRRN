import json
import numpy as np
from dltools.filelock.filelock import FileLock


class FileLogWriter(object):
    """
    This class let's you log arbitrary information to a log file at a specified location.
    """

    cache = []

    def __init__(self, filename, flush_frequency=2):
        """
        Initializes a new instance of the FileLogWriter class.

        :param filename: The filename of the log file.
        :param flush_frequency: The frequency at which the data shall be written to the log file.
        """

        # Open the file
        self.filename = filename
        self.flush_frequency = flush_frequency

    def _flush_cache(self):
        # Make sure that no one is currently reading the log file
        with open(self.filename, "a") as file_handle:
            with FileLock(self.filename):
                for entry in self.cache:
                    self._write_entry(file_handle, entry)

        self.cache = []

    def log(self, key, message):
        """
        Logs some information to the file
        :param key: The key under which the information shall be stored.
        :param message: The value that shall be logged.
        :param is_numpy: Whether or not the value is a numpy array.
        """

        # If message is a numpy array, then convert it to a python list first
        if type(message).__module__ == np.__name__:
            message = message.tolist()

        self.cache.append({"key": key, "value": message})
        if len(self.cache) > self.flush_frequency:
            self._flush_cache()

    def _write_entry(self, file_handle, entry):
        """
        Writes an entry to the log file.
        :param file_handle: The file handle of the log file.
        :param entry: The string to write to the log file
        """
        file_handle.write(json.dumps(entry))
        file_handle.write('\n')


class FileLogReader(object):
    """
    This class allows you to retrieve the information that is stored in a log file.
    """

    def __init__(self, filename):
        self.logs = {}
        self.filename = filename
        self.handle = None
        self.handle = open(self.filename, "r")

    def close(self):
        self.handle.close()

    def update(self):
        # Read the log messages
        with FileLock(self.filename):
            for row in self.handle:
                message = json.loads(row)

                # Does the key already exist?
                if message["key"] not in self.logs:
                    self.logs[message["key"]] = []

                self.logs[message["key"]].append(message["value"])
