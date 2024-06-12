#!/bin/bash

YEARS=1
MONTHS=0
INITIAL_POPULATION=10000
python src/scripts/profiling/scale_run.py \
--years $YEARS --months $MONTHS \
--initial-population $INITIAL_POPULATION \
--show-progress-bar

if [ $? -ne 0 ]
then
    echo "Python script failed"
    exit 125
else
    echo "Run complete"
fi
