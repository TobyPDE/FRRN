import os.path
import numpy as np
import uuid


class FileMutex(object):
    """
    This class implements a mutex that is organized in a file.
    """

    def __init__(self, filename):
        """
        Initializes a new instance of the file mutex.
        :type filename: The name of the mutex file.
        """
        self.filename = filename

    def __enter__(self):
        """
        Enters a critical section.
        """
        # Wait until the file does not exist
        while os.path.isfile(self.filename):
            # Noop
            pass

        # Create the file
        with open(self.filename, 'a'):
            os.utime(self.filename, None)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits a critical section.
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        try:
            # Delete the file
            os.remove(self.filename)
        except:
            pass


class AbstractFileLog(object):
    """
    This is the base class for all file loggers.
    """

    filename = ""
    """The name of the log file."""
    separator = ' '
    """This is the separator that is used in the log file"""

    def _get_mutex_filename(self):
        """
        Returns the name of the mutex file.
        :return: The name of the mutex file.
        """
        return self.filename + "__mutex__"


class FileLogWriter(AbstractFileLog):
    """
    This class let's you log arbitrary information to a log file at a specified location.
    """

    def __init__(self, filename):
        """
        Initializes a new instance of the FileLogWriter class.

        :param filename: The filename of the log file.
        """

        # Open the file
        self.filename = filename

    def log(self, key, value, is_numpy=False):
        """
        Logs some information to the file
        :param key: The key under which the information shall be stored.
        :param value: The value that shall be logged.
        :param is_numpy: Whether or not the value is a numpy array.
        """

        # Make sure that no one is currently reading the log file
        with FileMutex(self._get_mutex_filename()):
            with open(self.filename, "a") as file_handle:
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


class FileLogReader(AbstractFileLog):
    """
    This class allows you to retrieve the information that is stored in a log file.
    """

    def __init__(self, filename):
        """These are the logs. It's a map key -> [values,...]"""
        self.logs = {}
        self.filename = filename
        self.directory = os.path.split(filename)[0]

    def __enter__(self):
        self.file_handle = open(self.filename, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_handle.close()

    def update(self):

        # Read the log file
        #with FileMutex(self._get_mutex_filename()):
            # While we didn't reach the end of the file, keep on reading elements
            while True:
                try:
                    key = self._read_entry(self.file_handle)
                    value = self._read_entry(self.file_handle)

                    # Does the entry already exist?
                    if key not in self.logs:
                        self.logs[key] = []

                    self.logs[key].append(value)

                except ValueError:
                    # Stop reading the log file
                    break

    def _read_entry(self, file_handle):
        """
        Reads one entry from the file.

        :param file_handle: The file handle to the log file.
        :return: The read entry (np or text)
        """
        # Read the type of value that we expect
        # 0 for text, 1 for numpy
        value_type = file_handle.read(1)

        # Did we reach the end of the file?
        if len(value_type) == 0:
            raise ValueError("Reached EOF")

        # Read the length of the value
        value_length_as_str = ""
        while True:
            # Read the next character
            next_figure = file_handle.read(1)

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
        value = file_handle.read(value_length)

        # Do we need to convert it to numpy?
        if read_np:
            value = self.directory + "/" + os.path.basename(value)
            # Read the corresponding numpy array
            with np.load(value) as data:
                value = data['arr_0']

        return value
