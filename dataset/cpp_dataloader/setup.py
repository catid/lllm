from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}"
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(["cmake", "-S", ".", "-B", self.build_temp] + cmake_args)
        subprocess.check_call(["cmake", "--build", self.build_temp] + build_args)

setup(
    name="dataloader",
    ext_modules=[CMakeExtension("dataloader")],
    cmdclass={"build_ext": CMakeBuild}
)
