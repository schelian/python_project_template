FROM continuumio/anaconda3:latest
VOLUME "/host"
WORKDIR /host
CMD ["/bin/bash"]