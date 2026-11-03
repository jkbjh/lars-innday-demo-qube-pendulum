# Makefile for pip-compile command

# Define the output files
OUTPUT_FILE_CUDA = requirements.built.cuda
OUTPUT_FILE_CPU = requirements.built.cpu

# Define the input files
INPUT_FILES = pyproject.toml
INPUT_FILE_CUDA = requirements.cuda
INPUT_FILE_CPU = requirements.cpu

# Define the pip-compile command for CUDA and CPU
# PIP_COMPILE_CMD_CUDA = pip-compile -v --resolver=backtracking --no-upgrade $(INPUT_FILES) $(INPUT_FILE_CUDA) --output-file=requirements.built.cuda
# PIP_COMPILE_CMD_CPU = pip-compile -v --resolver=backtracking --no-upgrade $(INPUT_FILES) $(INPUT_FILE_CPU) --output-file=requirements.built.cpu
# --index="https://download.pytorch.org/whl/cpu" --default-index="https://pypi.org/simple"
PIP_COMPILE=uv pip compile --emit-index-url --extra dev
PIP_COMPILE_CMD_CUDA =$(PIP_COMPILE) $(INPUT_FILE_CUDA) $(INPUT_FILES)  --output-file=requirements.built.cuda
PIP_COMPILE_CMD_CPU = $(PIP_COMPILE)   $(INPUT_FILE_CPU) $(INPUT_FILES)  --output-file=requirements.built.cpu

.PHONY: all clean

# Default target to compile both CUDA and CPU requirements
all: requirements.built.cuda requirements.built.cpu

# Target for the CUDA requirements file
requirements.built.cuda: $(INPUT_FILES) $(INPUT_FILE_CUDA)
	$(PIP_COMPILE_CMD_CUDA)

# Target for the CPU requirements file
requirements.built.cpu: $(INPUT_FILES) $(INPUT_FILE_CPU)
	$(PIP_COMPILE_CMD_CPU)

# Clean target to remove the output files
clean:
	rm -f requirements.built.cuda requirements.built.cpu

