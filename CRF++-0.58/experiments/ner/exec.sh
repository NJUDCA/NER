#!/bin/sh
#../../crf_learn -f 3 -c 4.0 template train.data model > train.log
../../crf_test  -m model test_yw.data > result_yw

