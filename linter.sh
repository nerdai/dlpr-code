#!/bin/zsh

set -e
set -x

poetry run isort .
poetry run black .
poetry run flake8 .
poetry run pylint $(find ./dlpr_code -name "*.py" | xargs)
