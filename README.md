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

### Experimental Results of Gemini -
We have used Gemini pro 1.0. For more details, here is the [Gemini technical report](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)
###### Note : You may see "Gemini pro" in the code and test results, which is at the time of testing is an alias for Gemini pro 1.0

1. **Key-Value Retrieval Task for 75 Keys:**
   
| Correct Key at location -> | 0 | 24 | 49 | 74 |
| :---: | :---: | :---: | :---: | :---: |
| W/O QAC | 100% | 99.8% | 100% | 100% |
| W/ QAC | 100% | 100% | 100% | 100% |

2. **Key-Value Retrieval Task for 140 Keys:**
   
| Correct Key at location -> | 0 | 34 | 69 | 104 | 139 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| W/O QAC | 100% | 99.4% | 100% | 100% | 100% |
| W/ QAC | 99.8% | 100% | 100% | 100% | 100% |

3. **Key-Value Retrieval Task for 140 Keys:**
   
| Correct Key at location -> | 0 | 49 | 99 | 149 | 199 | 249 | 299 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| W/O QAC | 95.4% | 90% | 95.6% | 87% | 98.2% | 98.6% | 100% |
| W/ QAC | 97.2% | 100% | 100% | 100% | 100% | 99.8% | 100% |

4. **Question-Answer Task:**

| Relevant Document Location -> | 0 | 4 | 9 | 14 | 19 | 24 | 29 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| QA Task Closedbook | 43.61% | - | - | - | - | - | - |
| QA Task Oracle w/o QAC | 71.56% | - | - | - | - | - | - |
| QA Task Oracle w/ QAC | 78.30% |  - | - | - | - | - | - |
| QA Task on 10 Doc w/o QAC | 62.63% | 57.21% | 65.98% | - | - | - | - |
| QA Task on 10 Doc w/ QAC | 67.38% | 64.97% | 69.22% | - | - | - | - |
| QA Task on 20 Doc w/o QAC | 58.56% | 53.18% | 54.80% | 55.55% | 64.44% | - | - |
| QA Task on 20 Doc w/ QAC | 63.27% | 59.96% | 62.33% | 62.22% | 67.34% | - | - |
| QA Task on 30 Doc w/o QAC | 57.92% | 44.63% | 45.64% | 48.73% | 51.33% | 50.99% | 63.76 |
| QA Task on 30 Doc w/ QAC | 63.69% | 54.08% | 54.38% | 56.98% | 59.47% | 60.15% | 66.96% |


### Contributors -
1. [Anjali Chauhan](https://github.com/anjc24)
2. [Shishir Roy](https://github.com/yoR-rihsihS)
3. [Yash Patel](https://github.com/yash8071)
###### Check out commits to see exactly who contributed to what!
