# Installing `docker` on Linux

## Step 1:

It is recommended to uninstall any previous versions of `docker`.

```
sudo apt-get remove docker docker-engine docker.io containerd runc
```

## Step 2:

```
sudo apt-get update
```

## Step 3:

To set up a stable repository, run:

```
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo apt-key fingerprint 0EBFCD88
```

The output for `sudo apt-key fingerprint 0EBFCD88` should look like:

```
pub   rsa4096 2017-02-22 [SCEA]
      9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
uid           [ unknown] Docker Release (CE deb) <docker@docker.com>
sub   rsa4096 2017-02-22 [S]
```
Finally, to set up the repository, run:

```
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

## Step 4:

```
sudo apt-get update
```

## Step 5:

```
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

## Step 6:

Verify that the installation was successfull:

```
sudo docker run hello-world
```

A "Hello from Docker!" message should appear after the docker image was built.

## Step 7:

At the time this was written, the version should be 19.03.4.

```
docker --version
```

For more methods of installing `docker` like instlling specific versions go to the [official docker site](https://docs.docker.com/engine/install/ubuntu/).

# Using `docker` as a non-root user

In order to use `docker` as a non-root user, do the following steps. The rest of the `docker`-related commands in this repository will use `docker` as a non-root user. 

## Step 1:

```
sudo groupadd docker
```

## Step 2:

```
sudo usermod -aG docker $USER
```

Important note: "USER" in the above command does not have to be substituted with anything. 

## Step 3: 

```
newgrp docker
```

## Step 4: 

Verify that `docker` can be ran without sudo. This will download a test image, run the image, print out "Hello World!" and exit. 

```
docker run hello-world
```

For more information about using `docker` as a non-root user go to the [official docker site](https://docs.docker.com/install/linux/linux-postinstall/).
