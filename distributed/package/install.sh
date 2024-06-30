#!/bin/bash
pip install build

rm -rf build dist cpp_distributed.egg-info && pip uninstall cpp_distributed -y
python -m build && pip install --force-reinstall dist/*.whl
