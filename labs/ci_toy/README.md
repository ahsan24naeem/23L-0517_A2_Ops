# CI toy lab (Task 6.6)

Throwaway repository `23L-0517_A2_Ops-ci-toy`. Steps (run on your Mac):

```bash
cp -r labs/ci_toy ~/ci-toy && cd ~/ci-toy
git init -b main && git add . && git commit -m "toy CI: matrix 3.10/3.12"
git remote add origin https://github.com/<you>/23L-0517_A2_Ops-ci-toy.git && git push -u origin main   # GREEN run
# break it on purpose -> RED run
sed -i '' 's/return a + b/return a - b/' calc.py && git commit -am "bug: add subtracts" && git push
# fix it -> GREEN run
sed -i '' 's/return a - b/return a + b/' calc.py && git commit -am "fix add" && git push
```
