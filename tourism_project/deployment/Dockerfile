# Use a minimal base image with Python 3.10 installed
FROM python:3.11

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip3 install -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
        PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Add commands to check files and directory before running Streamlit
# The exec form of CMD does not run commands in a shell, so we use the shell form
# to chain commands with &&
CMD echo "Checking current directory and files before starting app..." && \
    pwd && \
    ls -lah && \
    echo "Attempting to run Streamlit app..." && \
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.enableXsrfProtection=false
