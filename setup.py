"""Setup script for building the C shared library."""
import os
import subprocess
import sys
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithLibrary(build_py):
    """Custom build command that compiles the C library."""

    def run(self):
        # Build the shared library
        src_dir = os.path.join(os.path.dirname(__file__), 'largekalman')

        # Determine output library name based on platform
        if sys.platform == 'darwin':
            lib_name = 'libfilter.so'  # macOS can use .so
            extra_flags = ['-dynamiclib']
        elif sys.platform == 'win32':
            lib_name = 'libfilter.dll'
            extra_flags = []
        else:
            lib_name = 'libfilter.so'
            extra_flags = ['-shared']

        lib_path = os.path.join(src_dir, lib_name)
        src_path = os.path.join(src_dir, 'libfilter.c')

        # Compile the library
        compile_cmd = [
            'gcc', '-O2', '-fPIC', '-Wno-unused-result',
            *extra_flags,
            '-o', lib_path,
            src_path,
            '-lm',  # Link math library
        ]

        print(f"Building {lib_name}...")
        print(f"Command: {' '.join(compile_cmd)}")

        try:
            subprocess.check_call(compile_cmd, cwd=src_dir)
            print(f"Successfully built {lib_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to build library: {e}")
            raise
        except FileNotFoundError:
            print("Error: gcc not found. Please install a C compiler.")
            print("  Ubuntu/Debian: sudo apt install build-essential")
            print("  macOS: xcode-select --install")
            print("  Fedora: sudo dnf install gcc")
            raise

        # Continue with normal build
        super().run()


setup(
    cmdclass={
        'build_py': BuildWithLibrary,
    },
    package_data={
        'largekalman': ['*.so', '*.dll', '*.dylib', '*.c'],
    },
)
