FROM ubuntu:16.04

# Install Dependencies
RUN apt-get update
RUN apt-get install -y build-essential bzip2 cmake curl git nano \
    nasm pkg-config rsync sudo tar timidity unzip wget \
    autoconf automake gcc g++ gfortran \
    libjpeg-dev \
    zlib1g-dev
#    software-properties-common \
#    libboost-all-dev \
#    libbz2-dev \
#    libfluidsynth-dev \
#    libfreetype6-dev \
#    libgme-dev \
#    libgtk2.0-dev \
#    libopenal-dev \
#    libpng12-dev \
#    libsdl2-dev \
#    libwildmidi-dev \
#    libzmq3-dev \
RUN apt-get clean all

# Build MPICH
#RUN mkdir /build/
#RUN cd /build && wget http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
#  && tar xvzf mpich-3.2.tar.gz && cd /build/mpich-3.2 \
#  && ./configure && make -j4 && make install && make clean && rm /build/mpich-3.2.tar.gz

# Python and Packages
RUN apt-get install -y python-dev python python-pip python-tk libvtk5-dev python-vtk
RUN pip install pip --upgrade
RUN pip install numpy
RUN pip install prody tqdm scipy scikit-image matplotlib sklearn \
keras tensorflow h5py opencv-python keras-vis mpi4py

# Enables X11 sharing and creates user home directory
ENV USER_NAME prot-struct-exp-cpu
ENV HOME_DIR /home/$USER_NAME
ENV HOST_UID 1000
ENV HOST_GID 1000
RUN export uid=${HOST_UID} gid=${HOST_GID} && \
    mkdir -p ${HOME_DIR} && \
    echo "$USER_NAME:x:${uid}:${gid}:$USER_NAME,,,:$HOME_DIR:/bin/bash" >> /etc/passwd && \
    echo "$USER_NAME:x:${uid}:" >> /etc/group && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME && \
    chown ${uid}:${gid} -R ${HOME_DIR}
USER ${USER_NAME}
WORKDIR ${HOME_DIR}

# Setup Bash Enviro
RUN mkdir src
RUN mkdir data
RUN echo 'PS1="\[$(tput bold)\]\[\033[38;5;10m\]\u@Docker\[$(tput sgr0)\]\[\033[38;5;15m\]:\[$(tput sgr0)\]\[\033[38;5;12m\]\W\[$(tput sgr0)\]\[\033[38;5;15m\]\\$\[$(tput sgr0)\]"' >> .bashrc
RUN ["/bin/bash"]
CMD 'bash'
