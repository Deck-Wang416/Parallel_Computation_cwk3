#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:t4:1

./cwk3 8
