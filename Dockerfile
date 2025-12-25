FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
# -----------------------------------------------------------------------------
# 基础环境配置
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
	LANG=zh_CN.UTF-8 \
	container=docker \
	TERM=xterm-256color

WORKDIR /workspace

COPY envs/*.sh /tmp/

RUN bash /tmp/install_base.sh

RUN bash /tmp/install_dev.sh

RUN bash /tmp/install_runtime.sh

RUN bash /tmp/install_graphics.sh

RUN bash /tmp/install_conda.sh 