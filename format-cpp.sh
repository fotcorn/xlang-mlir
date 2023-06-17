#!/usr/bin/env bash
find lib include calc-opt -name "*.h" -exec clang-format-16 -i {} \; 
find lib include calc-opt -name "*.cpp" -exec clang-format-16 -i {} \;
