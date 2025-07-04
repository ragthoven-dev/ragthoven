
<div align="center">
  <img alt="RAGthoven logo" src="images/ragthoven.png">
</div>

<br />

<div align="center">

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  <a href="https://pycqa.github.io/isort/"><img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" /></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
  ![Supported Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
  [![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ragthoven-dev/ragthoven?branch=master&label=Code%20Size&logo=GitHub&logoColor=ffffff&labelColor=282828&style=flat)]()
  [![GitHub repo size](https://img.shields.io/github/repo-size/ragthoven-dev/ragthoven?branch=master&label=Repo%20Size&logo=GitHub&logoColor=ffffff&labelColor=282828&style=flat)]()
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

</div>

<br />

# RAGthoven

<div align="center">

_**[`RAGthoven`](https://github.com/ragthoven-dev/ragthoven) is to Retrieval Augmented Generation (RAG) what [`axolotl`](https://github.com/axolotl-ai-cloud/axolotl) is to model finetuning.**_

</div>

Features:

- 🚀 Run end-to-end retrieval-augmented generation (RAG) experiments with minimal setup.
- ⚙️ Configure indexing, retrieval, re-ranking, and generation using YAML or CLI.
- ⚡ Processes the validation dataset in parallel, for lightning fast experiments.
- 🔍 Supports multiple retrievers and rerankers for scalable experimentation.
- 📊 Use matrixable options to define multiple pipeline configurations for batch evaluation.
- 📈 Automate benchmarking for consistent and reproducible RAG-based evaluations.
- 🛠️ Adapt and extend configurations for different NLP tasks.
- 🤖 Integrate with popular LLMs for generation and in-context learning.
- 📝 Log results for experiment tracking and evaluation.
- ...and more!

## Demo
https://github.com/user-attachments/assets/1bd037a7-0af5-4f93-9347-64143d74e5c6

## How to get started

### Prepare the repository

1. Install `poetry` (package manager for Python), following the [official instructions](https://python-poetry.org/docs/)
1. Install project dependencies:

    ```sh
    poetry install
    ```

1. Activate the virtual env:

    ```sh
    poetry shell
    ```

1. Create a `.env` file

    ```sh
    cp .env.sample .env
    ```

1. Set up the [LLM](https://docs.litellm.ai/docs/providers) by setting the correct environment variables.
   For instance, you can set your OpenAI API token to `OPENAI_API_KEY` in your `.env` file (see [.env.sample](./.env.sample)) or use ollama by providing `OLLAMA_API_BASE`.

### Run the project
1. First, follow the instructions in [Prepare the repository](#prepare-the-repository) section

1. Run the project

    ```sh
    ragthoven <config file path>
    ```
    Sample config files can be found in the `config/` directory.  
    
    For example, suppose we want to evaluate the example config `config/example_ag_news.yaml` whose contents are listed below:

    ```yaml
    name: "AG news"
    training_data:
      dataset: "fancyzhx/ag_news"
      input_feature: "text"
      label_feature: "label"
      split_name: "train"
      textual_labels: ["World", "Sports", "Business", "Sci/Tech"]
    validation_data:
      input_feature: "text"
      split_name: "test"
    results:
      output_filename: "example_ag_news_results"
    embed:
      k: 10
    rerank:
      k: 3
    llm:
      model: "azure/gpt-4o"
      sprompt: |
        You are a journalist. You would like to categorise each of the articles
        by it's headline. There are these 4 categories you can choose from:
        - World - news about world, general news
        - Sports - news about sports and related stuff
        - Business - news from business world
        - Sci/Tech - news about new science and technology
        As a busy journalist please answer with single world:
    
        Here are some of the headlines you labeled:
        {{ examples }}
      uprompt: |
        Please determine the category of the article based of this heading:
        {{ text }}
    ```

    As the configuration suggests, it is a text classification task, in which the article headlines are classified into four classes.
    In this particular case, we utilize both an embedding as well as the re-ranking mechanism to aid the LLM (in this case GPT-4o accessed via the Azure OpenAI Endpoints) with the classification.

    To obtain the actual results for the test split, you can run the following:

    ```sh
    ragthoven config/example_ag_news.yaml
    ```

3. Optional: Select output format

    You can also select the output format with the CLI option `--output`. See example below:
    
      ```sh
      ragthoven <config file path> --output csv
      ```
    
    The default output format is `.jsonl`, if the output format is not specified.
    
    You can always use the `--help` option to know which output formats are supported.

## YAML configuration

To configure RAGthoven to run experiment on a dataset it needs a valid configuration file. The structure of configuration file is logicaly split into sections and it is described bellow. The configuration file described below can bee found in [`config/comp_case2024-climate_matrix_multiprompt_custom_examples.yaml`](./config/comp_case2024-climate_matrix_multiprompt_custom_examples.yaml).

### Config file sections

- Every RAGthoven run requires a name, making it a required argument.
  ```yaml
  name: "Shared task on Climate Activism Stance and Hate Event Detection at CASE 2024"
  ```

- Training dataset is used to provide examples to the LLM. We need to specify the dataset source, textual intput, labels and split. 

  - `dataset` - This can be any huggingface dataset (eg. `nyu-mll/glue`), a `csv` file (as described bellow) or a `json` file. These can be specified by prefixing the path with `csv:` or `json:` respectively.
  - `input_feature` - which feature to use for embedding/reranking
  - `output_feature` (or `label_feature`) - which feature to use as `input_feature`'s classification label/regression score/summarization summary/translation target.
  - `split_name` - useful especially with HuggingFace dataset, leave as `train` when loading `json` or `csv`
  - `textual_labels` - some labels (especially in classification task) may come as indexes (0, 1, 2 ...). In order for LLM to make sense out of these labels, RAGthoven has the ability to translate indexes into textual labels. In the examples bellow this would mean `(0 => "No hatespeech", 1 => "Hate speech")`. Provide an empty array to pass the label directly.
  - `dataset_version` - (Optional) some dataset have multiple versions. Use this parameter to select the desired version.

  A full example:

  ```yaml
  training_data:
    dataset: "csv:./data/SubTask-A-train.csv"
    input_feature: "tweet"
    output_feature: "label"
    split_name: "train"
    textual_labels: ["No hatespeech", "Hate speech"]
    dataset_version: "001"
  ```

- Validation dataset is the primary source of input RAGthoven uses during its execution. This is the text that is used as query for retrieval/reranking and which is then *judged* by LLM.
  - `dataset` - (Optional) the same applies as for `training_data.dataset`. When not provided `training_data.dataset` is used instead.
  - `input_feature` - what feature is used for retrieval/reranking. This feature can be passed into prompts as `{{ text }}`
  - `split_name` - the same applies as for `training_data.split_name`
Here is a full example:

  ```yaml
  validation_data:
    dataset: "csv:./data/SubTask-A-test.csv"
    input_feature: "tweet"
    split_name: "train"
  ```

- Output specific configurations
  - `output_cached` - (Optional) when running validation on huge datasets (eg. 30k validation data)  in case of a crash (e.g. due to no API credits available ...) only partial data is written into the results file by default. When running RAGthoven once again, the results file is recreated and all existing results data is lost. In order to prevent this, it is possible to set `output_cached` to `true`. When running again RAGthoven loads results file and performs inference only on `ids` that are not yet present in the results file.
  - `output_cache_id` - (Optional) specify what is the name of the feature to be used as `id` in the output
  - `output_filename` - (Optional) specify the output file name (whithout extension)
  - `bad_request_default_value` - (Optional) what is the desired output of the RAGthoven when the request ends in error (e.g. because the LLM API refuses to respond...)
A full example:

  ``` yaml
  results:
    output_cached: true
    output_cache_id: "index"
    output_filename: "results"
    bad_request_default_value: -1
  ```

- Retrieval is a crucial part of RAG. It is configured in the  next section. When retrieval is not used, you may omit this configuration section entirely (see `ragthoven/config/comp_case2024-climate_matrix_no_retrieval.yaml`)
  - `k` - number of relevant training samples to be retrieved for every query. *Matrixable* parameter, provide an array of values (eg. `[20, 10, 5]`) to run in *matrix* mode.
  - `training_size_limit` - (Optional) when doing testing, it is best to not start by embedding all training samples, as this can be time consuming. Setting this option reduces training dataset to more manageable number.
  - `model` - (Optional) choose any HuggingFace-provided model for retrieval. Basic retrieval is done  using [Chroma](https://github.com/chroma-core/chroma. *Matrixable*, specify models in an array.
A full example can be seen below:

  ```yaml
  embed:
    k: 20
    training_size_limit: 20
    model: "sentence-transformers/all-MiniLM-L6-v2"
    embedder: "sbert"
    device: "cpu"
    docs_embedding_count: 512
  ```

- In order to obtain high performance on the retrieved results, it is often a good idea to rerank them and select only the most relevant ones. This is done in cross encoder fashion. When reranking is not used, omit this configuration entry completely (see `ragthoven/config/comp_case2024-climate_matrix_no_rerank.yaml`)
  - `k` - select only top-`k` reranked from retrieved samples. *Matrixable*, specify `k`s in an array.
  - `training_size_limit` - limits number of documents used for retrieval. First number of documents is used.
  - `model` - (Optional) choose model for reranking. Basic reranker utilizes [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank/), which you can consult for more details on available models. *Matrixable*, specify models in an array.
A full example can be seen below:
  - `embedder` - currently there are two embedding methods available, `sbert` which utilizes `sentence-transformers` models supported by `Chromadb` and `cde` which can use `Contextual Document Embeddings (CDE)` models, eg. `jxm/cde-small-v1`. In both cases `chromadb` is used as vector database.
  - `device` - when utilizing `sbert` as embedder, this option allows for setting the device used in the embedding computation (eg. `cuda`, defaults to `cpu`)
  - `docs_embedding_count` - used only in `cde` mode. Number of documents to use for creating dataset embeddings. Randomly sampled. See [https://huggingface.co/jxm/cde-small-v1](https://huggingface.co/jxm/cde-small-v1) for more information. When `docs_embedding_count` > `training_size_limit` then `docs_embedding_count` = `training_size_limit`. 

  ```yaml
  rerank:
    k: [5, 1]
    model: "ms-marco-MiniLM-L-12-v2"
  ```

- At the heart of RAGthoven, there is always a call to an LLM API, be it local or commercial.
  - `model` - which model to use. *Matrixable*, provide an array of models.
  - `temperature` - (Optional) - temperature for the models decoder. Defaults to `0`. *Matrixable*, provide array with temperature values.
  - `examples` - (Optional) examples can be ordered, arranged and arbitrary textual information can be added. The `text` and `label` are as proved in configuration for the training dataset. Other than that, a variable `examples[i].data.feature`, where `feature` is a feature present in the original dataset allows for access of any feature for the given example.

  A full example can be seen below:

  ```yaml
  llm:
    model: "gpt-4o"
    temperature: 0
    examples: |
      This is example number one: {{ examples[0].text }} and it's corresponding label: {{ examples[0].label }}
      ...
      This is example number two: {{ examples[10].text }} and it's corresponding label: {{ examples[10].label }}
      If you want more examples than there are retrieved/reranker, it will fail and throw an exception!
  ...
  ```

### Special features
#### Matrix configuration

Some parameters are *Matrixable* - support matrix configuration. The matrix configuration runs for every configuration.

eg. when two variables are set `k: [20, 10]` and `n: [5, 1]` the there are 4 combinations evaluated:
1. `k: 20, n: 5`
2. `k: 20, n: 1`
3. `k: 10, n: 5`
4. `k: 10, n: 1`

adding another variable with two options to the matrix would result in 8 combinations.

#### Multiple prompts

  Prompts can be specified in two ways: 
  - **Multiprompt version**: specify prompts in array named `prompts` with following structure:
    - `name` - name of the prompt, this is used to later access output of specific prompt as input in another. There is one special prompt added into every call called `system` with role `system`.
    - `role` - select `system` or `user` role (or any other role supported by the model).
    - `prompt` - jinja2 formatable prompt. Each prompt is provided with `examples`, `text` variables, output of previous propts accessible by `{{ previous_prompt_name.out }}` and `{{ data.<feature> }}` where `<feature>` is a feature that is available in validation dataset.
  - **Single prompt version** (see `ragthoven/config/comp_case2024-climate_matrix_with_custom_examples.yaml`)
    - `sprompt` - single system prompt. *Matrixable*, provide an array of prompts.
    - `uprompt` - single user prompt. *Matrixable*, provide an array of prompts.
    both prompts can make use of `{{ examples }}`, `{{ text }}` and `{{ data.<feature> }}` where `<feature>` is a feature that is available in validation dataset.

An example of the **multiprompt** version of the configuration.
  ```yaml
  ...
    prompts: 
      -
        name: "system"
        role: "system"
        prompt: |
          Analyze the input tweet to determine if it is hate speech or not, based on the following criteria:
          
          1. Some instructions here
          ...
          4. Some instructions there

          ## Examples
        
          {{ examples }}
      -
        name: "select_key_phrases"
        role: "user"
        prompt: |
          Try to reason and think of the words and phrases that best describe the mood/behaviour of the person writing
          that text. Try to list at least 5 - 6 words/phrases.
          Text: {{ text }}
      -
        name: "result"
        role: "user"
        prompt: |
          Please determine the category of the text given the text and words/phrases describing it's sentiment.
          Use "1" for hatespeech tweets and "0" for all other:
          Text: {{ text }}
          Sentiment keywords: {{ select_key_phrases.out }}
          ANSER ONLY WITH A SINGLE NUMBER!
  ```

An example of the **single prompt** version of the configuration.

  ```yaml
  ...
    sprompt: |
      Analyze the input tweet to determine if it is hate speech or not, based on the following criteria:

      1. Some instructions here
      ...
      4. Some instructions there

      ## Examples

      {{ examples }}
    uprompt: |
      Please determine the category of the text use "1" for hatespeech tweets and "0" for all other:
      {{ text }}
      ANSER ONLY WITH SINGLE NUMBER!
  ```

#### Preprocessing
RAGthoven provides a way to run custom python code on every example in the validation dataset. Please refer to code [`ragthoven/tools/example_tool.py`](ragthoven/tools/example_tool.py) on how to write a tool for RAGthoven preprocessing. In order to specify which tools to run, specify them in `yaml` config as follows: 

(Note that the tools are run sequentially, in the order of appearance in the `entries` key of the YAML config.)

```yaml
preprocessor:
  entries: ["example_tool.fizzbuzz", "example_tool.count_ands"]
```

#### Function calling
RAGthoven provides a way to use function calling. LLMs can be provided with tools (e.g. functions to call) which can help them fetch fresh data from various sources or take actions (e.g. call an API, send an email etc). The LLM can then decide on its own whether it needs to call any of the provided functions and what arguments to pass to those functions. You can read more about function calling [here](https://platform.openai.com/docs/guides/function-calling?example=get-weather). 

Note that the LLM does not actually perform the function call - it merely notes in its response that it wants the function to be executed with specific arguments, and then RAGthoven executes the function. Finally, the results of the function call can be used in the 2 following ways:
1. If you set `llm.messages: true` in the config, then the results of the function call will be forwarded to the LLM in any api call to the LLM for any future prompts (for the current validation example). The messages are passed along as part of the array of messages which contain the whole conversation, that the LLM API accepts.
2. If you set `llm.messages: false` in the config, then no messages array is passed to the LLM and the results of the function call can be used manually in a subsequent prompt to the LLM. You need to specify in the yaml config where they will be used inside a prompt.

Please refer to code [`ragthoven/tools/example_fun_calling.py`](ragthoven/tools/example_fun_calling.py) on how to write functions for function calling. Also, in order to use function calling, please use multiprompt in your yaml config (example: [`config/single-shot-example-function-calling.yaml`](config/single-shot-example-function-calling.yaml)). In order to use this example please install `wikipedia` package:

```yaml
llm:
  ...
  tools: ["example_fun_calling.WikipediaPageSearch", "example_fun_calling.WikipediaPageSummary"]
  prompts:
    -
      name: "system"
      role: "system"
      prompt:
        You are the best at knowing ...
    -
      name: "wikipedia_search"
      role: "user"
      tools: ["WikipediaPageSearch"]
      prompt: |
        First, let's have a look at wikipedia page about this movie. This is the text of the review:
        {{ data.text }}

        Please first find some useful information online about this movie.
    -
      name: "wikipedia_summary"
      role: "user"
      tools: ["WikipediaPageSummary"]
      prompt: |
        Now, you have obtained following list of results for your search:
        {{ wikipedia_search.out }}
        Please obtain a summary of this movie.
    ...
```


## Tests

In order to run tests:

1. First, follow the instructions in [Prepare the repository](#prepare-the-repository) section

2. Run tests (uses only 10 data entries for embedding and runs on 2 validation data
   entries):

    ```sh
    make test
    ```

## Tutorials

### News headlines classification
This is a tutorial about how to use RAGthoven to classify News headlines into categories.

1. First, follow the instructions in [Prepare the repository](#prepare-the-repository) section. This takes care of installing the relevant dependencies, activating the virtual env, and preparing a `.env` file with the relevant env variables needed to interact with the LLM API of your choice.

1. In order to use RAGthoven, you need to specify your RAG pipeline and any other parameters in a yarml configuration file. 

    For this tutorial, we are going to use an existing example yaml configuration file available [here](ragthoven/config/example_ag_news.yaml).

1. Let's explore some parts of the yaml file we will be using (open the file from the previous point to follow along if you like):
    1. Under the section `training_data` we are specifying `dataset: "fancyzhx/ag_news"`, which is a Hugging Face dataset containing news articles headlines and their respective labels. We will be using this dataset to populate our Vector database with training examples, which will be retrieved at inference time and included in the LLM prompt as examples for in-context learning.
    1. Under the `results` section we are specifying `output_filename: "example_ag_news_results"`. RAGthoven will use the string `example_ag_news_results` as the prefix to the output files it generates. 
    1. Under the `embed` section we are specifying `training_size_limit: 10`, which means that we will insert only the 10 first examples from the dataset into our Vector databse, and they will be available for querying at inference time.
    1. Under the `rerank` section we are specifying `k: 3`. This means that, at inference time, we will rerank the retrieved training examples from the Vector database and include the 3 best matches in our LLM prompt.
    1. Under the section `llm` we are specifying various parameters related to the generation step of the RAG pipeline. For example, we can specify the system and user prompts that we are passing to the LLM api (`sprompt` and `uprompt` respectively).

1. Run RAGthoven against the yaml file:

    ```sh
    ragthoven config/example_ag_news.yaml
    ```

    Note that this step will take a lot of time to run, since it needs to process the whole validation dataset. Feel free to terminate the program prematurely after about 20 seconds to move on to the next step of this tutorial.

1. RAGthoven should have created 2 output files. The names of both those files have the prefix `example_ag_news_results`. One of the files has the suffix `.metadata.json` and it is a metadata file that contains all the parameters used for that RAG pipeline run. The second file has the suffix `.jsonl` and contains the output of the classification task for some of the validation dataset examples.

## License

RAGthoven is licensed under the terms of the MIT license.

## Paper & Citation

You can find more info on RAGthoven, as well as a few case studies, in the associated paper: https://aclanthology.org/2025.coling-demos.12/

If you found RAGthoven helpful in your work, please consider citing it:

```
@inproceedings{karetka-etal-2025-ragthoven,
    title = "{RAG}thoven: A Configurable Toolkit for {RAG}-enabled {LLM} Experimentation",
    author = "Karetka, Gregor  and
      Skottis, Demetris  and
      Dutkov{\'a}, Lucia  and
      Hra{\v{s}}ka, Peter  and
      Suppa, Marek",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven  and
      Mather, Brodie  and
      Dras, Mark",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics: System Demonstrations",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-demos.12/",
    pages = "117--125"
}
```
