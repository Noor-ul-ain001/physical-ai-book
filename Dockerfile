FROM nvidia/cudagl:12.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install pip -U
RUN pip3 install -r requirements.txt

# Copy project files
COPY . .

# Install Isaac Sim
# This is a simplified version - actual installation would be more complex
RUN git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
RUN cd IsaacGymEnvs && pip3 install -e .

# Install Docusaurus dependencies
RUN npm install

# Build the documentation
RUN npm run build

# Expose port for Docusaurus
EXPOSE 3000

# Default command
CMD ["npm", "start"]
