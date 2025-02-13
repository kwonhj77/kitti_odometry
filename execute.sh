#!/bin/
python main.py --train --save_results stacked_1024 --test --save_checkpoint stacked_1024 --params stacked_1024 | tee ./stacked_1024.txt
python main.py --train --save_results stacked_512 --test --save_checkpoint stacked_512 --params stacked_512 | tee ./stacked_512.txt
python main.py --train --save_results stacked_256 --test --save_checkpoint stacked_256 --params stacked_256 | tee ./stacked_256.txt