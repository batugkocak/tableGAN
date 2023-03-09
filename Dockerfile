FROM tensorflow/tensorflow:1.15.5-jupyter 

COPY requirements.txt /tf/

RUN python -m pip install --upgrade pip
RUN pip install -r /tf/requirements.txt


# docker build -t mytablegan ./
# winpty docker run -it --name tableGan-test -v "C:\\Users\\stavg\\Documents\\DLPadmin\\":/tf/notebooks \
# -p 8888:8888 mytablegan
# winpty docker restart tableGan-test
#
# * pls see commands in testing_tableGan.ipynb
# * they *should* work
