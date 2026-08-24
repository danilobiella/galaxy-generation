GREEN := \033[1;32m
RESET := \033[0m

RAW_DATA_FILE := data/Galaxy10_DECals.h5
BIN_DATA_FILES := data/galaxy10_decals_lmdb

KAGGLE_DATASET_STAGING := .kaggle_dataset
KAGGLE_STAGING := .kaggle_push

API_HOST := 127.0.0.1
API_PORT := 8000

DOCKER_IMAGE := galaxy-api
HF_SPACE := galaxy-generation

.PHONY: data kaggle-dataset-stage kaggle-dataset-push kaggle-dataset-clean \
        kaggle-stage kaggle-push kaggle-clean api api-cpu \
        docker-build docker-run hf-check hf-deploy

data: $(BIN_DATA_FILES)
	@echo "$(GREEN)Data preparation complete!$(RESET)"

# Download raw data
$(RAW_DATA_FILE):
	@echo "$(GREEN)Downloading raw data...$(RESET)"
	mkdir -p data 
	wget -c https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5 -O $(RAW_DATA_FILE)

# Process raw data into binary format
$(BIN_DATA_FILES): $(RAW_DATA_FILE) scripts/process_data.py
	@echo "$(GREEN)Processing raw data into binary format...$(RESET)"
	python scripts/process_data.py

# Upload dataset on kaggle
kaggle-dataset-stage: kaggle-dataset-clean $(BIN_DATA_FILES)
	@echo "$(GREEN)Staging files for Kaggle...$(RESET)"
	mkdir -p $(KAGGLE_DATASET_STAGING)
	cp dataset-metadata.json $(KAGGLE_DATASET_STAGING)/
	cp $(BIN_DATA_FILES)/* $(KAGGLE_DATASET_STAGING)/

# Push dataset to Kaggle
kaggle-dataset-push: kaggle-dataset-stage
	@echo "$(GREEN)Pushing dataset to Kaggle...$(RESET)"
	kaggle datasets create -p $(KAGGLE_DATASET_STAGING)

kaggle-dataset-clean:
	@echo "$(GREEN)Cleaning dataset directory...$(RESET)"
	rm -rf $(KAGGLE_DATASET_STAGING)

kaggle-stage: kaggle-clean
	@echo "$(GREEN)Staging files for Kaggle...$(RESET)"
	kaggle-bundler build

kaggle-push: kaggle-stage
	@echo "$(GREEN)Pushing to Kaggle...$(RESET)"
	kaggle kernels push -p $(KAGGLE_STAGING)

kaggle-clean:
	@echo "$(GREEN)Cleaning staging directory...$(RESET)"
	kaggle-bundler clean

# Serve the model
api:
	@echo "$(GREEN)Starting API on $(API_HOST):$(API_PORT)...$(RESET)"
	uvicorn api:app --host $(API_HOST) --port $(API_PORT)

# Same, forced onto CPU (what the deployed container will look like)
api-cpu:
	@echo "$(GREEN)Starting API on CPU at $(API_HOST):$(API_PORT)...$(RESET)"
	JAX_PLATFORMS=cpu uvicorn api:app --host $(API_HOST) --port $(API_PORT)

# Build the inference image exactly as Hugging Face will
docker-build:
	@echo "$(GREEN)Building $(DOCKER_IMAGE)...$(RESET)"
	docker build -t $(DOCKER_IMAGE) .

docker-run: docker-build
	@echo "$(GREEN)Running $(DOCKER_IMAGE) on :7860...$(RESET)"
	docker run --rm -p 7860:7860 $(DOCKER_IMAGE)
