# nlp-nebula
## A repository containing code for **DS207: Introduction to NLP** (IISc, Bangalore) course project. 
###### Repository name has nothing to do with the project topic.

Original Paper = "Lost in the Middle: How Language Models Use Long Contexts".
Read paper [here](https://arxiv.org/pdf/2305.13048.pdf).

### Objective -
To find out whether LLMs can really utilize their full context length.

### Experiment/Task Descriptions -
1. **Key-Value Retrieval :**
Given $k$ (unique) key-value pairs and a (one) key, find its corresponding value, where $k \in \{75, 140, 300\}$. The position of the query key is varied and its effect on the accuracy is noted.
2. **Multi-Document Question-Answer :**
Given $k$ documents (text excerpts) and a question, give the answer, where $k \in \{10, 20, 30\}$. Moreover, exactly one of the k documents is relevant for answering the question and remaining are distractors. The position of the relevant document is varied and its effect on the accuracy is noted.
###### These are just high level overview of the experiments. A detailed description will be added latter.

### Contributors -
1. [Anjali Chauhan](https://github.com/anjc24)
2. [Shishir Roy](https://github.com/yoR-rihsihS)
3. [Yash Patel](https://github.com/yash8071)
###### Check out commits to see exactly who contributed to what!
