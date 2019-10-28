# Installing `nvidia-docker` on Linux

Check [prerequisites]() before installing `nvidia-docker`. 

## Step 1:

Install the `nvidia-docker` repository by executing the following commands in order. More information is available [here](https://nvidia.github.io/nvidia-docker/).

```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  
sudo apt-get update
```

## Step 2:

```
sudo apt-get install nvidia-docker2
```

## Step 3:

```
sudo pkill -SIGHUP dockerd
```

## Step 4:

Verify that the installation was successfull.

```
docker run --gpus=all --rm nvidia/cuda nvidia-smi
```

For more installation methods go [here](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)).