#!/bin/bash

python main.py --train --save_train_results batch32 --test --save_test_results batch32 --save_checkpoint batch32 --params batch32
python main.py --train --save_train_results batch64 --test --save_test_results batch64 --save_checkpoint batch64 --params batch64
python main.py --train --save_train_results batch256 --test --save_test_results batch256 --save_checkpoint batch256 --params batch256