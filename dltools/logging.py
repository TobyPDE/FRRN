import os.path
import numpy as np
import uuid


class FileLogWriter(object):
    """
    This class let's you log arbitrary information to a log file at a specified location.
    """

    cache = []

    def __init__(self, filename):
        """
        Initializes a new instance of the FileLogWriter class.

        :param filename: The filename of the log file.
        """

        # Open the file
        self.filename = filename
        self.separator = ' '

    def _flush_cache(self):
        # Make sure that no one is currently reading the log file
        with open(self.filename, "a") as file_handle:
            for entry in self.cache:

                key, value, is_numpy = entry

                self._write_entry(file_handle, key, np_flag="0")

                if is_numpy:
                    # Create a unique filename and store the numpy thing in a separate file
                    id = uuid.uuid1()
                    np_file_name = self.filename + str(id) + ".npz"

                    # Save the numpy array in the file
                    np.savez(np_file_name, value)

                    # Store the filename in the logs
                    self._write_entry(file_handle, np_file_name, np_flag="1")
                else:
                    # Simply write the shit to the file
                    self._write_entry(file_handle, value, np_flag="0")
        self.cache = []

    def log(self, key, message, is_numpy=False):
        """
        Logs some information to the file
        :param key: The key under which the information shall be stored.
        :param message: The value that shall be logged.
        :param is_numpy: Whether or not the value is a numpy array.
        """

        self.cache.append([key, message, is_numpy])
        if len(self.cache) > 100:
            self._flush_cache()

    def _write_entry(self, file_handle, entry, np_flag="0"):
        """
        Writes an entry to the log file.
        :param file_handle: The file handle of the log file.
        :param entry: The string to write to the log file
        """
        file_handle.write(np_flag)
        file_handle.write(str(len(str(entry))))
        file_handle.write(self.separator)
        file_handle.write(str(entry))


class FileLogReader(object):
    """
    This class allows you to retrieve the information that is stored in a log file.
    """

    def __init__(self, filename):
        self.logs = {}
        self.filename = filename
        self.separator = ' '

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update(self):

        # Read the log file
        with open(self.filename, "r") as handle:
            content = handle.readlines()

        # While we didn't reach the end of the file, keep on reading elements
        index = 0
        while index < len(content):
            try:
                key, index = self._read_entry(content, index)
                value, index = self._read_entry(content, index)

                # Does the entry already exist?
                if key not in self.logs:
                    self.logs[key] = []

                self.logs[key].append(value)

            except ValueError:
                # Stop reading the log file
                break

    def _read_entry(self, content, index):
        """
        Reads one entry from the file.

        :param content: The log file
        :param index: The current cursor position
        :return: The read entry (np or text)
        """
        # Read the type of value that we expect
        # 0 for text, 1 for numpy
        value_type = content[index]
        index += 1

        # Did we reach the end of the file?
        if len(value_type) == 0:
            raise ValueError("Reached EOF")

        # Read the length of the value
        value_length_as_str = ""
        while True:
            # Read the next character
            next_figure = content[index]
            index += 1

            # If this is a white-space, then we are done reading the length
            if next_figure == self.separator:
                break
            else:
                # Simply add it to the length
                value_length_as_str += next_figure

        # Convert the read elements to logic values
        read_np = value_type == "1"
        value_length = int(value_length_as_str)

        # Read the raw value
        value = content[index:index + value_length]
        index += value_length

        # Do we need to convert it to numpy?
        if read_np:
            # Read the corresponding numpy array
            with np.load(value) as data:
                value = data['arr_0']

        return value, index
