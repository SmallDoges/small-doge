# SmallDoge WebUI Extensions - Dataset and Training Management

This document describes the extensions added to SmallDoge WebUI to support dataset management and model training workflows as requested in issue #26.

## Overview

The SmallDoge WebUI has been extended with comprehensive dataset and training management capabilities. The implementation follows the suggested approach of using FastAPI to wrap utility functions as APIs and Gradio for frontend pages.

## Architecture

### Backend API Extensions

The backend now includes two new routers:

#### Dataset Management API (`/api/v1/datasets`)

**Endpoints:**
- `GET /available` - List available datasets for download
- `GET /downloaded` - List already downloaded datasets  
- `POST /download` - Download a specific dataset
- `POST /download-multiple` - Download multiple datasets
- `GET /info/{dataset_name}` - Get dataset information
- `DELETE /delete/{dataset_name}` - Delete a downloaded dataset

**Supported Datasets:**
- **Pretrain**: fineweb-edu, cosmopedia-v2, finemath
- **Finetune**: smoltalk, ultrafeedback_binarized, open_thoughts

#### Training Management API (`/api/v1/training`)

**Endpoints:**
- `GET /types` - Get available training types
- `GET /models` - Get available model architectures
- `GET /configs/{model_type}` - Get configurations for model type
- `POST /start` - Start a training job
- `GET /status/{job_name}` - Get training job status
- `GET /jobs` - List all training jobs
- `GET /logs/{job_name}` - Get training logs
- `POST /stop/{job_name}` - Stop a training job

**Supported Training Types:**
- **pretrain** - Language model pretraining
- **sft** - Supervised fine-tuning
- **dpo** - Direct Preference Optimization
- **grpo** - Group Relative Policy Optimization

**Supported Model Architectures:**
- **doge** - Original SmallDoge architecture
- **doge2** - Enhanced SmallDoge architecture

### Frontend Extensions

A new management interface has been created (`management_app.py`) that provides:

#### Dataset Management Tab
- Browse available datasets by type (pretrain/finetune)
- Download datasets with configurable paths and processing
- View downloaded datasets
- Dataset information and management

#### Training Management Tab  
- Configure training parameters (type, model, dataset path, output directory)
- Start training jobs with proper configuration
- Monitor training job status
- View training logs and manage running jobs

#### Information Tab
- Comprehensive documentation
- Usage instructions
- API endpoint reference
- Dataset and training type descriptions

## File Structure

```
src/small_doge/webui/backend/smalldoge_webui/
├── utils/
│   ├── dataset_utils.py       # Dataset download and management utilities
│   └── training_utils.py      # Training job management utilities
└── routers/
    ├── datasets.py            # Dataset management API routes
    └── training.py            # Training management API routes

src/small_doge/webui/frontend/
└── management_app.py          # Dataset and training management UI
```

## Usage

### Option 1: Management Interface (Recommended)

Launch the dedicated management interface:

```bash
# Method 1: Direct execution
python -m src.small_doge.webui.frontend.management_app

# Method 2: Using the CLI (when available)
small-doge-webui --management
```

This starts a dedicated interface at `http://localhost:7862` with tabs for:
- 📊 Datasets - Download and manage datasets
- 🎯 Training - Configure and monitor training jobs  
- 📖 Information - Documentation and usage guide

### Option 2: API Direct Access

Start the backend and use the API directly:

```bash
# Start backend
python -m src.small_doge.webui.backend.start

# API available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### Option 3: Original Chat Interface

The original chat interface remains unchanged and can be used as before:

```bash
small-doge-webui
# Available at http://localhost:7860
```

## Example Workflows

### Download and Use a Dataset

1. **Download a dataset:**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/datasets/download" \
        -H "Content-Type: application/json" \
        -d '{"dataset_name": "fineweb-edu", "save_dir": "./data/datasets", "cache_dir": "./data/cache"}'
   ```

2. **Use via Management Interface:**
   - Open http://localhost:7862
   - Go to Datasets tab
   - Select dataset and download location
   - Click "Download Dataset"

### Start a Training Job

1. **Via API:**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/training/start" \
        -H "Content-Type: application/json" \
        -d '{
          "training_type": "pretrain",
          "model_type": "doge", 
          "dataset_path": "./data/datasets/fineweb-edu",
          "output_dir": "./data/training/my-model",
          "model_config": "./recipes/doge/Doge-160M/config_full.yaml"
        }'
   ```

2. **Via Management Interface:**
   - Open http://localhost:7862
   - Go to Training tab
   - Configure training parameters
   - Click "Start Training"
   - Monitor progress in the jobs section

## Implementation Notes

### Backend Utilities

- **dataset_utils.py**: Wraps existing dataset download scripts from `examples/utils/`
- **training_utils.py**: Provides training job management and status tracking
- Both modules handle errors gracefully and provide detailed status information

### Frontend Design

- Clean, intuitive interface using Gradio components
- Real-time status updates and progress feedback
- Comprehensive error handling and user guidance
- Responsive design that works across different screen sizes

### Integration Approach

The implementation extends the existing WebUI without breaking changes:
- Original chat interface remains fully functional
- New APIs are additive (no existing endpoints modified)
- New frontend runs on separate port to avoid conflicts
- Modular design allows selective usage of features

## API Documentation

Full API documentation is available when the backend is running:
- Interactive docs: http://localhost:8000/docs
- ReDoc format: http://localhost:8000/redoc

## Error Handling

The system includes comprehensive error handling:
- Dataset download failures with detailed error messages
- Training configuration validation
- Network connectivity checks
- File system permission validation
- Process management for training jobs

## Future Enhancements

Potential improvements for future versions:
- Real-time training progress monitoring
- Distributed training support
- Model performance metrics dashboard
- Dataset preprocessing pipelines
- Automated hyperparameter tuning
- Integration with experiment tracking systems

## Conclusion

The SmallDoge WebUI has been successfully extended with comprehensive dataset and training management capabilities. The implementation provides both API and UI access to all requested functionality while maintaining compatibility with the existing system.

Users can now easily:
- Download and manage datasets
- Configure and start training jobs
- Monitor training progress
- Deploy and test models

This addresses all requirements specified in issue #26 and provides a solid foundation for future AI workflow automation.