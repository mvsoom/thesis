#!/usr/bin/env bash
# activate the venv if it’s here
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi
# then strip outputs
nbstripout