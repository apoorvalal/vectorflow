from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import subprocess
import os
import shutil


class BuildDLibrary(build_py):
    def run(self):
        # Build the D library
        print("Building D shared library with dub...")
        # Since pyvectorflow is now inside the vectorflow repo, the root is just one level up
        vectorflow_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        # Ensure D environment is set up (assuming user has ldc2 in path or sourced dacti)
        # We can try to source it if we know where it is, but better to assume environment is ready
        # or use the provided activation script logic if strictly necessary.
        # For now, we assume 'dub' and 'ldc2' are in the PATH.

        # We need to construct the command.
        # If running in a shell that has dlang activated, simple `dub build` works.
        # But `subprocess` might not inherit aliases.

        # Try to find dub
        dub_exe = shutil.which("dub")
        if not dub_exe:
            # Fallback for the specific environment we are in
            dub_exe = os.path.expanduser("~/dlang/ldc-1.41.0/bin/dub")

        ldc2_exe = shutil.which("ldc2")
        if not ldc2_exe:
            ldc2_exe = os.path.expanduser("~/dlang/ldc-1.41.0/bin/ldc2")

        if not os.path.exists(dub_exe):
            raise RuntimeError(
                "Cannot find 'dub' executable. Please install D compiler or activate it."
            )

        cmd = [
            dub_exe,
            "build",
            "--root",
            vectorflow_dir,
            "--config=shared-lib",
            "--compiler=" + ldc2_exe,
            "--build=release",
            "--force",
        ]

        subprocess.check_call(cmd)

        # Copy the built library to the package directory
        lib_name = "libvectorflow.so"
        src_lib = os.path.join(vectorflow_dir, lib_name)
        dst_lib = os.path.join(self.build_lib, "vectorflow", lib_name)

        print(f"Copying {src_lib} to {dst_lib}")
        self.mkpath(os.path.dirname(dst_lib))
        shutil.copyfile(src_lib, dst_lib)

        # Continue with standard build_py
        super().run()


setup(
    name="pyvectorflow",
    version="0.2.0",
    description="Python bindings for Vectorflow (Ctypes)",
    packages=find_packages(),
    cmdclass={
        "build_py": BuildDLibrary,
    },
    include_package_data=True,
    package_data={
        "vectorflow": ["*.so"],
    },
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: D",
    ],
)
