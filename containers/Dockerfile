FROM quay.io/condaforge/miniforge3

WORKDIR /opt

COPY .MouseCHD /root/.MouseCHD

RUN git clone https://github.com/hnguyentt/MouseCHD && \
    cd MouseCHD && \
    conda install python=3.9 && \
    conda install pytorch torchvision pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia -c conda-forge && \
    conda install -c conda-forge jupyter && \
    pip install --no-cache-dir "tensorflow[and-cuda]==2.14.0" && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir "napari[all]" && \
    cd .. && \
    rm -rf MouseCHD && \
    conda clean -a -y && \
    apt-get autoremove && \
    apt-get clean && \
    pip cache purge && \
    rm -rf /var/lib/apt/lists/* /opt/conda/pkgs/* /root/.cache/pip && \
    export PATH=$PATH:/opt/conda/bin && \
    export LD_LIBRARY_PATH=/opt/conda/lib/:/opt/conda/lib/python3.9/site-packages/nvidia/cudnn/lib