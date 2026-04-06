GREEN := \033[1;32m
RESET := \033[0m

RAW_DATA_FILE := data/Galaxy10_DECals.h5
BIN_DATA_FILES := data/galaxy10_decals_lmdb

.PHONY: data

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


