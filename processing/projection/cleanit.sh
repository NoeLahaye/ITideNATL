#!/bin/bash

fils="outjob_*.[oe]* proj_*_ty-loop.[0-9]*.py job_proj_?_[0-9]*.sh"

if [ $# -eq 1 ] && [ $1 = "rm" ]; then
    rm -f $fils
else
    mv $fils out_files/
fi

