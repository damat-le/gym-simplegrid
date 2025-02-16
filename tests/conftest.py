import pytest
import gymnasium
import numpy as np
import matplotlib

def pytest_report_header():
    """Print version information in the header of the test report."""
    out_ =  f"------------------" + '\n' + \
            f"gymnasium: {gymnasium.__version__}" + '\n' + \
            f"numpy: {np.__version__}" + '\n' + \
            f"matplotlib: {matplotlib.__version__}" + '\n' + \
            f"------------------"
    return out_
