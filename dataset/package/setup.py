from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys
import multiprocessing

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
        ]
        build_args = ["--config", "Release"]

        num_threads = multiprocessing.cpu_count() - 1
        if num_threads > 1:
            build_args.append(f"-j{num_threads}")

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(["cmake", "-S", ".", "-B", self.build_temp] + cmake_args)
        subprocess.check_call(["cmake", "--build", self.build_temp] + build_args)

setup(
    name="cpp_dataloader",
    version="0.1.0",
    python_requires='>=3',
    ext_modules=[CMakeExtension("cpp_dataloader")],
    cmdclass={"build_ext": CMakeBuild},
    package_data={
        'cpp_dataloader': ['cpp_dataloader_library.so'],
    },
    include_package_data=True,
    package_dir={'': 'python_src'},
    packages=find_namespace_packages(where='python_src'),
    install_requires=["numpy<2.0"]
)
