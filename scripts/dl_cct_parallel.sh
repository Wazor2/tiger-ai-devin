#!/bin/bash
# High-throughput CCT image download using parallel wget from a URL list.
set -e
cd /home/ubuntu/project
python3 scripts/make_url_list.py
echo "url list ready: $(wc -l < /home/ubuntu/project/datasets/url_list.txt)"
export -f dl_one 2>/dev/null
cd /home/ubuntu/project/datasets/cct_subset
cat ../url_list.txt | xargs -P 16 -I {} wget -q --timeout=45 --tries=2 -nv -O /dev/null {} >/dev/null 2>&1 || true
# The above -O /dev/null wastes bandwidth; use awk instead.
