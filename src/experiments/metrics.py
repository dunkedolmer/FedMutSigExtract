# metrics.py

def accuracy(correctly_identified_signatures: int, total_actual_signatures: int) -> float:
    """
    Calculates and returns the accuracy of a mutational signature extraction method.
    
    The accuracy is defined as the number of correctly identified signatures divided by the total number of actual signatures.

    Args:
        correctly_identified_signatures (int): The number of correctly identified mutational signatures.
        total_actual_signatures (int): The total number of actual mutational signatures.

    Returns:
        float: The accuracy as a float between 0 and 1.

    Raises:
        ValueError: If total_actual_signatures is 0 to avoid division by zero.
    """
    if total_actual_signatures == 0:
        raise ValueError("The total number of actual signatures must be greater than 0.")
    return correctly_identified_signatures / total_actual_signatures

def execution_time(time_start: float, time_end: float) -> float:
    """
    Calculates and returns the execution time of a method.

    The execution time is the difference between the end time and the start time.

    Args:
        time_start (float): The start time of the execution in seconds.
        time_end (float): The end time of the execution in seconds.

    Returns:
        float: The total execution time in seconds.

    Raises:
        ValueError: If time_end is less than time_start.
    """
    if time_end < time_start:
        raise ValueError("The end time must be greater than or equal to the start time.")
    return time_end - time_start

def communication_overhead(total_bytes_sent: int, total_bytes_received: int) -> int:
    """
    Calculates and returns the total communication overhead in bytes.

    The communication overhead is the sum of the total bytes sent and received.

    Args:
        total_bytes_sent (int): The total number of bytes sent.
        total_bytes_received (int): The total number of bytes received.

    Returns:
        int: The total communication overhead in bytes.
    """
    return total_bytes_sent + total_bytes_received
