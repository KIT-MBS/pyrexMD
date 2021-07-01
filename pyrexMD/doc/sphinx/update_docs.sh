#!/usr/bin/bash

rm -rf _build/*
make html
firefox _build/html/index.html
