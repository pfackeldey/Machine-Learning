FROM riga/law_example_base

# labels
LABEL name="law_example_singletop"
LABEL version="0.0.1"

# workdor
WORKDIR /root/law_example

# basic environment variables
ENV LAW_SANDBOX docker::riga/law_example_singletop

# python packages
RUN pip install numpy
RUN pip install scipy
RUN pip install matplotlib
RUN pip install numexpr
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install root_numpy
RUN pip install scinum
RUN pip install order
RUN pip install seaborn
RUN pip install tensorflow
RUN pip install keras
RUN pip install pyyaml
