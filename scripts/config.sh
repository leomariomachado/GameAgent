apt update -y
apt list --upgradable
apt install -y rsync cmake build-essential ninja-build gdb -y
apt-get install -y elfutils libdw-dev
apt-get install -y elfutils libdw-dev libelf-dev
apt-get install ffmpeg imagemagick -y 
apt-get install unzip -y
python -m pip install --upgrade pip
kaggle install
pip install kaggle
pip3 install kaggle
export KAGGLE_API_TOKEN=KGAT_ba7eb5a1a4202e30b4a96107891b6d6f
apt-get update && apt-get install -y p7zip-full
pip install gdown
apt install -y nsight-systems-2025.6.3
apt-mark unhold libnccl2 libnccl-dev
apt install -y \
  libnccl2=2.30.4-1+cuda12.9 \
  libnccl-dev=2.30.4-1+cuda12.9
apt-mark hold libnccl2 libnccl-dev
dpkg -l | grep nccl

watch -n 1 nvidia-smi

rsync -avz --delete \
-e "ssh -p 24664 -i ~/.ssh/id_ed25519" \
~/CLionProjects/BaseAI/ \
root@213.173.110.202:/tmp/BaseAI/

rsync -avz --delete \
-e "ssh -p 24664 -i ~/.ssh/id_ed25519" \
~/CLionProjects/GameAgent/ \
root@213.173.110.202:/tmp/GameAgent/
