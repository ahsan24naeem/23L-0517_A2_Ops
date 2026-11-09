#!/usr/bin/env bash
# Task 6.1 - Docker warm-up with a ready-made image (nginx:alpine). Run on your Mac with Docker Desktop.
# Screenshot each numbered step.
set -x
cd "$(dirname "$0")"
mkdir -p site && echo "<h1>Hello from a mounted volume - 23L-0517 $(date)</h1>" > site/index.html

docker pull nginx:alpine                                    # 1. get an image
docker run -d --name vg-warmup -p 8080:80 \
  -v "$PWD/site":/usr/share/nginx/html:ro nginx:alpine      # 2. run a container, map port 8080->80, mount a volume
docker ps                                                   # 3. list running containers
curl -s http://localhost:8080                               # 4. port mapping works (served from the mounted folder)
echo "<p>edited on the host, visible without restarting</p>" >> site/index.html
curl -s http://localhost:8080
docker logs vg-warmup | tail -5                             # 5. read its logs
docker exec vg-warmup sh -c 'whoami; hostname; ls /usr/share/nginx/html; cat /etc/os-release | head -2'  # 6. shell
# interactive version for the screenshot:  docker exec -it vg-warmup sh
docker image ls nginx                                       # 7. list images
docker history nginx:alpine | head -8                       #    the layers of the image
docker stop vg-warmup && docker rm vg-warmup                # 8. stop + remove the container
docker ps -a | grep vg-warmup || echo "container removed"
docker rmi nginx:alpine                                     # 9. remove the image
docker image ls nginx
