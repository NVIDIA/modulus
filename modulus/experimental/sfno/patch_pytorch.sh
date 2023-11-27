#!/bin/bash

SRC_FILE=./third_party/torch/distributed/utils.py
PYTHON_DIST_PKG_DIR=$(python -c "import site; print(site.getsitepackages()[0])")

if cp ${SRC_FILE} ${PYTHON_DIST_PKG_DIR}/torch/distributed/; then
    echo "Patching complete"
else
    echo "Patching failed, possibly permissions issue, trying with sudo..."
    if [ -n "$(which sudo)" ]; then
        if sudo cp ${SRC_FILE} ${PYTHON_DIST_PKG_DIR}/torch/distributed/; then
            echo "Patching complete"
        else
            echo "cp failed, giving up."
        fi
    else
        echo "sudo was not found, giving up."
    fi
fi
