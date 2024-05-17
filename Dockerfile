# Use Python 3.10.12 as the base image
FROM python:3.10.12

# Set the working directory
WORKDIR /IOH-Profiler-HDBO-Comparison

# Print the working directory for debugging
RUN pwd

# Update the package list and upgrade pip
RUN apt-get update && \
    pip install --upgrade pip

# Copy files into the container
COPY . .

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

RUN unzip bayes_optim.zip
RUN unzip skopt.zip
RUN unzip Bayesian-Optimization.zip
RUN unzip GPy.zip
RUN unzip GPyOpt.zip
RUN unzip RDUCB.zip

RUN mv bayes_optim /usr/local/lib/python3.10/site-packages/
RUN rm -rf /usr/local/lib/python3.10/site-packages/skopt*
RUN mv skopt /usr/local/lib/python3.10/site-packages/
RUN mv GPy /usr/local/lib/python3.10/site-packages/
RUN mv GPyOpt /usr/local/lib/python3.10/site-packages/
RUN mv Bayesian-Optimization mylib/lib_BO_bayesoptim
RUN mv RDUCB mylib/lib_RDUCB/HEBO