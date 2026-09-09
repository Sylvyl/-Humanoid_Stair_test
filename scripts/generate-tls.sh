#!/usr/bin/env bash
set -euo pipefail
dest="${1:-$HOME/x2dev/stair-dashboard/tls}"
mkdir -p "$dest"
umask 077
openssl req -x509 -newkey rsa:3072 -sha256 -days 365 -nodes \
  -keyout "$dest/key.pem" -out "$dest/cert.pem" \
  -subj "/CN=10.0.1.41" \
  -addext "subjectAltName=IP:10.0.1.41,DNS:agi-3.local"
openssl x509 -in "$dest/cert.pem" -noout -fingerprint -sha256
echo "Verify this fingerprint from the PC2 terminal before trusting the certificate."

