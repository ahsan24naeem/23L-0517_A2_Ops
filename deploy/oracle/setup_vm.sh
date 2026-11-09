#!/usr/bin/env bash
# One-time setup of an Oracle Cloud "Always Free" Ubuntu VM (Ampere A1 arm64 or E2.1.Micro x86).
# Run on the VM:  bash setup_vm.sh
set -euo pipefail
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git netfilter-persistent
# Docker Engine + Compose plugin (official convenience script)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
# Oracle's Ubuntu images ship an iptables policy that REJECTs everything except SSH.
# This is the VM's own firewall - separate from the VCN Security List in the console.
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save
# Small swap file helps on the 1 GB E2.1.Micro shape (TensorFlow import needs ~600 MB)
if ! swapon --show | grep -q swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi
echo "Done. Log out and back in (docker group), then clone the repo and copy models/champion.keras."
