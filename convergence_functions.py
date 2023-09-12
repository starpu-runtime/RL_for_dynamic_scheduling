from collections import deque

import numpy as np

from common_logging import training_logger


class ConvergenceFunctions:
    def __init__(self, required_buffer_elements, convergence_threshold):
        self.required_buffer_elements = required_buffer_elements
        self.execution_performance_buffer = deque(maxlen=self.required_buffer_elements)
        self.convergence_threshold = convergence_threshold
        self.rolling_max = 0

    def average(self, performance):
        if len(self.execution_performance_buffer) < self.required_buffer_elements:
            return False

        average = np.average(self.execution_performance_buffer)
        diff = np.abs(performance - average)
        converged = diff < self.convergence_threshold

        training_logger.info("Checking if model has converged")
        training_logger.info(f"Performance buffer: {self.execution_performance_buffer}")
        training_logger.info(
            f"Current performance: {performance}, Average: {average}, Threshold: {self.convergence_threshold}"
        )
        training_logger.info(f"Has converged? {converged}")

        return converged

    def dynamic_rolling_maximum(self, _):
        if len(self.execution_performance_buffer) < self.required_buffer_elements:
            return False

        # Calculate rolling maximum
        self.rolling_max = max(self.execution_performance_buffer)

        # Count the number of performances that are close to the rolling maximum
        close_to_max_count = 0
        required_high_performances = int(self.required_buffer_elements * 0.9)  # At least 90% should be high

        for perf in reversed(self.execution_performance_buffer):
            if perf >= self.rolling_max * 0.95:  # Within 5% of the rolling maximum
                close_to_max_count += 1
                if close_to_max_count >= required_high_performances:
                    return True
            else:
                close_to_max_count = 0  # Reset if a value is not close to max

        return False

    def append_buffer(self, data):
        self.execution_performance_buffer.append(data)
        training_logger.info(
            f"Inserting in buffer. Current performance: {data}, Buffer: {self.execution_performance_buffer}"
        )
