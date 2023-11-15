FROM pytorch/pytorch:latest

ADD SPANet /opt/SPANet

RUN  cd /opt/SPANet && pip install -e .

# Install the required Python packages
RUN pip install \
    numpy \
    sympy \
    scikit-learn \
    numba \
    opt_einsum \
    h5py \
    cytoolz \
    tensorboardx \
    seaborn \
    rich \
    pytorch-lightning==1.7

CMD ["bash"]
