#!/usr/bin/env bash
# scripts/vsnb-clean.sh
# read .vsnb JSON on stdin, drop outputs/executionSummary + empty-value cells

jq '
  # remove the output blobs
  del(.cells[].outputs, .cells[].executionSummary)
  |
  # then keep only cells whose .value is non-empty
  .cells |= map(select(.value != ""))
'
