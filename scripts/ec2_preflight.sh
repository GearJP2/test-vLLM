#!/usr/bin/env bash
# Validate an EC2 GPU host before downloading model weights or starting vLLM.
set -euo pipefail

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

command -v nvidia-smi >/dev/null || fail "nvidia-smi is missing; use an NVIDIA-ready AMI or install a compatible driver"
command -v docker >/dev/null || fail "docker is missing"
docker compose version >/dev/null || fail "Docker Compose v2 is missing"

printf '%s\n' 'GPU:'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
printf '%s\n' 'Docker:'
docker version --format '{{.Server.Version}}'

available_gib=$(df -Pk . | awk 'NR == 2 { print int($4 / 1024 / 1024) }')
if [ "$available_gib" -lt 40 ]; then
  fail "only ${available_gib} GiB free in the repository filesystem; use a 100 GiB root volume and retain at least 40 GiB after the vLLM image is pulled for a 7B smoke test"
fi
printf 'Available disk in repository filesystem: %s GiB\n' "$available_gib"

[ -f .env ] || fail "copy .env.example to .env and set the pinned vLLM image digest before continuing"
grep -q 'REPLACE_WITH' .env && fail ".env still contains placeholder values"
docker compose config >/dev/null || fail "compose configuration is invalid"

printf '%s\n' 'EC2 host preflight passed.'
