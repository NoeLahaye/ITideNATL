#!/bin/bash

fils="outjob_*.[oe]* proj_pres_ty-loop*.[0-9]*.py"

if [ $# -eq 1 ] && [ $1 = "rm" ]; then
    rm -f $fils
else
    mv $fils out_files/
fi

