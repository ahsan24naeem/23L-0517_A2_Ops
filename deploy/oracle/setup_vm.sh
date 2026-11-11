#!/usr/bin/env bash
# One-time setup of a cloud VM for the VisionGuard model API.
# Works on Ubuntu 22.04/24.04 (AWS EC2, Oracle Cloud; user ubuntu) and Oracle Linux 8/9 (user opc).
# Run on the VM:  bash deploy/oracle/setup_vm.sh
set -euo pipefail
. /etc/os-release
echo "OS: $PRETTY_NAME ($(uname -m)), user: $USER"

if command -v dnf >/dev/null; then
  # ---- Oracle Linux ----
  sudo dnf install -y -q dnf-plugins-core git rsync
  sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
  sudo dnf install -y -q docker-ce docker-ce-cli containerd.io docker-compose-plugin
  sudo systemctl enable --now docker
  # The VM's own firewall (separate from the VCN Security List in the console)
  if command -v firewall-cmd >/dev/null; then
    sudo firewall-cmd --permanent --add-port=8000/tcp
    sudo firewall-cmd --reload
    sudo firewall-cmd --list-ports
  fi
else
  # ---- Ubuntu ----
  sudo apt-get update -y
  sudo apt-get install -y ca-certificates curl git rsync
  curl -fsSL https://get.docker.com | sudo sh
  # Oracle's Ubuntu images REJECT everything except SSH in iptables; AWS images have no such rule
  # (on AWS only the Security Group filters traffic). Opening 8000 explicitly is harmless on both.
  if sudo iptables -S INPUT | grep -q REJECT; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y netfilter-persistent
    sudo iptables -I INPUT 1 -p tcp --dport 8000 -m state --state NEW -j ACCEPT
    sudo netfilter-persistent save
  fi
fi
sudo usermod -aG docker "$USER"

# Swap: the 1 GB E2.1.Micro shape needs it for TensorFlow (import alone ~600 MB)
if ! swapon --show | grep -q /swapfile; then
  sudo dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
  sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
free -h | head -3

# The container runs as an unprivileged user (uid 10001): the mounted model must be world-readable
if [ -d "$HOME/23L-0517_A2_Ops/models" ]; then chmod -R a+rX "$HOME/23L-0517_A2_Ops/models"; fi
docker --version || sudo docker --version
echo "Done. Log out and back in (docker group), then start the API."
