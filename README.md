# Sheikh Mujibur Rahman Bangla NLP Dataset

This dataset contains Bangla text related to Sheikh Mujibur Rahman (known as "Bangabandhu"), the founding father of Bangladesh. The data is collected from two primary sources:

1. **Prothom Alo** - One of Bangladesh's leading Bangla newspapers
2. **Bangla Wikipedia** - The Bangla language version of Wikipedia

## Dataset Description

The dataset is designed for natural language processing (NLP) tasks on Bangla text, particularly for tasks related to historical figures, political discourse, and Bangladeshi history.

### Contents

- **News Articles**: Articles from Prothom Alo related to Sheikh Mujibur Rahman
  - Title, URL, summary, full content, and publication date
  - Filtered to ensure relevance to the subject

- **Wikipedia Content**: Structured information from the Bangla Wikipedia page about Sheikh Mujibur Rahman
  - Section-by-section content
  - References
  - Image metadata

## Data Format

The dataset is available in multiple formats:

1. **JSON**: Complete dataset with all collected information
   - `mujib71_data.json` - Contains both news articles and Wikipedia data

2. **CSV**: News articles in tabular format
   - `news_articles.csv` - Easy to use for data analysis and model training

3. **Metadata**: Information about the dataset itself
   - `metadata.json` - Version, description, sources, creation date, and counts

## Data Collection Methodology

The data was collected using a Python web scraper that:

1. Searches Prothom Alo for articles containing "শেখ মুজিবুর রহমান" (Sheikh Mujibur Rahman)
2. Extracts content from the Bangla Wikipedia page dedicated to Sheikh Mujibur Rahman
3. Filters and processes the content to ensure relevance and quality
4. Organizes the data into machine-readable formats

The scraper respects website rate limits and employs polite scraping practices.

## Dataset Enhancement with LLMs

This dataset is enhanced using large language models (LLMs) to provide additional high-quality content:

- **Generated Content**: Additional examples that complement the scraped data
- **Multi-turn Conversations**: Synthetic chat examples about Sheikh Mujibur Rahman
- **Instruction Examples**: Task-specific examples in the Alpaca format
- **Pretraining Text**: Additional Bangla text for continued pretraining

The enhancement process uses:

1. **Gemini API**: Google Gemini 1.5 Pro for generating comprehensive Bangla content
2. **GROQ API**: Alternative to use Llama 3 for content generation

To customize the enhancement process:

```bash
# Basic usage with default settings (10 examples per dataset type)
python enhance_dataset_with_llm.py

# Specify model and number of examples
python enhance_dataset_with_llm.py --model gemini --examples 5

# Generate only for specific dataset type
python enhance_dataset_with_llm.py --dataset-type chat --model groq
```

## Potential Uses

- Training Bangla language models on historical and political text
- Named entity recognition for Bangladeshi historical figures
- Sentiment analysis on text about political leaders
- Information extraction about historical events
- Development of Bangla NLP tools with historically significant content

## Citation

If you use this dataset in your research or applications, please cite it as:

```
@dataset{mujib71_dataset,
  author       = {Likhon Sheikh},
  title        = {Sheikh Mujibur Rahman Bangla NLP Dataset},
  month        = may,
  year         = 2025,
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/likhonsheikhdev/mujib71}}
}
```

## Acknowledgments

- Data sourced from [Prothom Alo](https://www.prothomalo.com) and [Bangla Wikipedia](https://bn.wikipedia.org/)
- Images included in the dataset metadata are credited to their original sources on Wikipedia

## License

This dataset is provided for research and educational purposes. The original content remains under the copyright of the respective sources.

## Contact

For questions or feedback about this dataset, please contact the dataset creator through Hugging Face.
