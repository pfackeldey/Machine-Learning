FROM centos:7

# labels
LABEL name="base"
LABEL version="0.0.1"

# workdor
WORKDIR /root/law_example

# basic environment variables
ENV LAW_SANDBOX docker::mfackeld/base
ENV PATH /usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV LIBPATH /usr/local/lib:$LIBPATH
ENV PYTHONPATH /usr/local/lib:/usr/local/lib64/python2.7/site-packages:$PYTHONPATH
ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
ENV CMAKE_MODULE_PATH /usr/local/etc/cmake
ENV MANPATH /usr/local/man

# prepare yum
RUN yum -y update; yum clean all
RUN yum -y install yum-plugin-priorities; yum clean all

# basic software
RUN yum install -y epel-release; yum clean all
RUN yum groupinstall -y "Development Tools"; yum clean all
RUN yum install -y gcc-gfortran openssl-devel pcre-devel mesa-libGL-devel mesa-libGLU-devel \
    glew-devel ftgl-devel fftw-devel cfitsio-devel libxml2-devel libldap-dev libpng-devel \
    avahi-compat-libdns_sd-devel gsl-static gcc-c++ gcc atlas atlas-devel lapack-devel blas-devel \
    curl-devel json-c-devel git make cmake binutils swig nano htop which screen wget python-devel \
    python-pip gfal2-all gfal2-devel gfal2-python; yum clean all

# additional software
RUN mkdir /software

# gfal2 dropbox plugin
RUN git clone https://github.com/cern-it-sdc-id/gfal2-dropbox.git && \
    cd gfal2-dropbox && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr && make && make install && \
    cd ../.. && rm -rf gfal2-dropbox

# root
ENV ROOTSYS /software/root
RUN cd /software && \
    wget -nv https://root.cern.ch/download/root_v6.10.08.Linux-centos7-x86_64-gcc4.8.tar.gz && \
    tar -xzf root_v6.10.08.Linux-centos7-x86_64-gcc4.8.tar.gz && \
    rm root_v6.10.08.Linux-centos7-x86_64-gcc4.8.tar.gz && \
    echo "source \$ROOTSYS/bin/thisroot.sh" >> /etc/profile.d/software.sh

# python packages
RUN pip install requests
RUN pip install six
RUN pip install luigi
