# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automating various tasks in the Sheikh Mujibur Rahman Bangla NLP Dataset project.

## Available Workflows

### 1. Dataset Update (`dataset_update.yml`)

Automatically updates and uploads the dataset to HuggingFace.

**Triggers:**
- Push to main branch
- Weekly schedule (Sunday at 02:00 UTC)
- Manual trigger (workflow_dispatch)

**Process:**
1. Runs web scraper to collect data
2. Downloads relevant images
3. Creates HuggingFace Chat dataset format
4. Uploads both datasets to HuggingFace repositories
5. Updates README with timestamp

**Requirements:**
- `HUGGINGFACE_TOKEN` secret must be set in repository settings

### 2. Continuous Integration (`ci.yml`)

Checks code quality and runs tests for all changes.

**Triggers:**
- Push to any branch
- Pull requests to main branch

**Process:**
1. Lints code with flake8
2. Runs unit tests with pytest
3. Collects test coverage
4. Reports coverage to Codecov (if configured)

### 3. Documentation (`docs.yml`)

Generates and publishes project documentation.

**Triggers:**
- Push to main branch (only when README.md or docs/** files change)
- Manual trigger (workflow_dispatch)

**Process:**
1. Creates MkDocs configuration
2. Generates documentation from README and additional docs
3. Builds and publishes to GitHub Pages

## Configuration

To configure these workflows:

1. Add the `HUGGINGFACE_TOKEN` secret:
   - Go to repository Settings > Secrets and variables > Actions
   - Create a new repository secret named `HUGGINGFACE_TOKEN`
   - Value should be a HuggingFace API token with write permissions

2. Enable GitHub Pages:
   - Go to repository Settings > Pages
   - Set source to "GitHub Actions"

## Manual Triggers

You can manually trigger workflows from the Actions tab in the GitHub repository.

## Customization

Edit the YAML files in the `.github/workflows/` directory to customize the workflow behavior:
- Adjust schedule times
- Modify build configurations
- Add or remove steps as needed
