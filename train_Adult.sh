#!/usr/bin/env bash
echo "Adult Data Sets"
python3 main.py --train --dataset=Adult --epoch=100 --test_id=OI_11_00
