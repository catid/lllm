#!/bin/bash
pip install build

rm -rf build dist cpp_dataloader.egg-info && pip uninstall cpp_dataloader -y
python -m build && pip install --force-reinstall dist/*.whl
