FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

ENV LANG C.UTF-8

EXPOSE 8000

WORKDIR /app

COPY . /app/

RUN pip uninstall -r uninstall.txt && pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt 

ENTRYPOINT ["sh", "./run.sh"]