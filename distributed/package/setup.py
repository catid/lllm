from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys
import multiprocessing

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class BuildPackage(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # Ensure we have Rust installed
        try:
            subprocess.check_call(['rustc', '--version'])
        except:
            raise RuntimeError("Rust must be installed to build quiche")

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Build quiche
        quiche_dir = os.path.join(script_dir, 'quiche')
        print(f"Building quiche in {quiche_dir}")
        subprocess.check_call(['cargo', 'build', '--release', '--features', 'ffi'], cwd=quiche_dir)

        # Copy the built library to your project directory
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        print(f"Copying quiche library to {extdir}")
        subprocess.check_call(['cp', os.path.join(quiche_dir, 'target/release/libquiche.a'), extdir])

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
        ]
        build_args = ["--config", "Release"]

        num_threads = multiprocessing.cpu_count() - 1
        if num_threads > 1:
            build_args.append(f"-j{num_threads}")

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(["cmake", "-S", ".", "-B", self.build_temp] + cmake_args, cwd=script_dir)
        subprocess.check_call(["cmake", "--build", self.build_temp] + build_args, cwd=script_dir)

setup(
    name="cpp_distributed",
    version="0.1.0",
    python_requires='>=3',
    ext_modules=[CMakeExtension("cpp_distributed")],
    cmdclass={"build_ext": BuildPackage},
    package_data={
        'cpp_distributed': ['cpp_distributed_library.so', 'libquiche.a', 'libcuSZp.a'],
    },
    include_package_data=True,
    package_dir={'': 'python_src'},
    packages=find_namespace_packages(where='python_src'),
    install_requires=["numpy<2.0"]
)
