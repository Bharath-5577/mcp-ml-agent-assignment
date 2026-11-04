# MCP Assignment — Build and Call Your Own MCP Tool

## 1. How many tools does the server offer now? What changed in the client?

The MCP server now offers **9 tools**, compared to the original 8. I added a custom tool called `predict_both`, which runs Logistic Regression and KNN on the same patient data and checks whether their predictions agree.

I also created a new client, `my_client.py`, that discovers the available tools, trains both models, and calls `predict_both` for all 114 patients in the test dataset.

## 2. How many patients did the models disagree on? Which model was correct?

The models disagreed on **8 out of 114 patients**.

The disagreement patient indices were **3, 25, 48, 53, 76, 80, 99, and 102**.

| Patient | Actual diagnosis | Logistic Regression | KNN | Correct model |
|---|---|---|---|---|
| 3 | Benign | Benign | Malignant | Logistic Regression |
| 25 | Benign | Benign | Malignant | Logistic Regression |
| 48 | Benign | Benign | Malignant | Logistic Regression |
| 53 | Malignant | Benign | Malignant | KNN |
| 76 | Malignant | Malignant | Benign | Logistic Regression |
| 80 | Benign | Benign | Malignant | Logistic Regression |
| 99 | Benign | Benign | Malignant | Logistic Regression |
| 102 | Malignant | Malignant | Benign | Logistic Regression |

## 3. Which model performed better on the disagreement cases?

**Logistic Regression performed better**, correctly predicting 7 of the 8 disagreement cases. KNN correctly predicted only 1.

This shows that Logistic Regression was more reliable for the cases where the models produced different predictions. However, KNN correctly identified patient 53 as malignant when Logistic Regression predicted benign, showing that neither model was perfect.

## 4. Why is the tool's docstring important for an AI application?

A docstring explains what a tool does and helps an AI application understand when and how to use it. In MCP, tool descriptions are exposed to clients during tool discovery.

For my `predict_both` tool, the docstring explains that it runs both models and checks whether they agree. This makes the tool easier for an AI agent to select and use correctly.

## 5. Why is the Parquet audit log useful?

The Parquet audit log records information about MCP tool calls, including the request ID, tool name, and execution status.

It is useful for debugging, monitoring, and verifying that tools executed successfully. It also provides a record of what happened during the experiment.

I verified the audit log using pandas. The final entries showed successful calls to `get_sample_input` and `predict_both` for patients 109 through 113, confirming that the last test cases were processed and recorded.