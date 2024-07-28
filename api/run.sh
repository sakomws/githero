#!/usr/bin/env bash
pipenv run uvicorn main:app --host 0.0.0.0 --port 10001 --workers 8