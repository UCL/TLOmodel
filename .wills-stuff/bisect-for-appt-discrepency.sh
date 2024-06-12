#!/bin/bash

# Good commit: 012ef6d, checksum should be dc273b2e0ef27a3d61afb954ede48b8ba038a9a7
REFERENCE_CHECKSUM=dc273b2e0ef27a3d61afb954ede48b8ba038a9a7
YEARS=0
MONTHS=1
INITIAL_POPULATION=1000
RUN_CHECKSUM=$( \
    python src/scripts/profiling/scale_run.py \
    --years $YEARS --months $MONTHS --log-final-population-checksum \
    --initial-population $INITIAL_POPULATION | tail -n 1 \
    | grep -oP "(?<=checksum: ).*(?=\"])" \
)

if [ $? -ne 0 ]
then
    echo "Python script failed"
    exit 125
fi

echo "Run checksum was:  $RUN_CHECKSUM"
echo "Checksum to match: $REFERENCE_CHECKSUM"

if [[ "$RUN_CHECKSUM" == "$REFERENCE_CHECKSUM" ]]
then
    echo "That's a match"
else
    echo "No match, fail"
    exit 1
fi
