"""Keep Windows from idle-sleeping while a long training or search job runs.

The hold belongs to this process and ends when it exits. A closed lid still sleeps.
No-op on other operating systems.
"""
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def hold():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
