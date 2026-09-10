# Capstone Report — Applied Search Intelligence

* **Author:** Thahasin Jamal
* **Lane:** Applied Search Intelligence — Content Opportunity Scoring
* **Repo:** FlyRank ML Internship repository
* **Date:** September 2026

## 0. Abstract

This project asks whether a simple machine learning model can identify content with a higher measured likelihood of decline and help a content team decide what to review first. The analysis uses 30,000 anonymized content records containing search-performance, content, freshness, and engagement signals. A Logistic Regression model using five features was compared with a simple rule-based baseline using the same evaluation data. The model achieved an F1 score of 0.681 on the main random split and 0.648 on a stricter client-grouped validation, with a strong improvement in recall over the baseline. The final output is a ranked decision-support queue that groups content into REFRESH, REVIEW, and MONITOR actions for human review.

## 1. Problem framing

The decision supported by this work is:

**Which content should a team review first?**

The unit of analysis is an individual content item.

The model produces a decline probability that can be used to rank content by priority. The ranked output is then grouped into three action categories:

1. **REFRESH** — highest-priority content for investigation.
2. **REVIEW** — medium-priority content requiring further inspection.
3. **MONITOR** — lower-priority content that can be watched without immediate action.

The human editor remains responsible for the final decision.

A wrong call has two main costs. Prioritizing healthy content can waste limited editorial time, while missing genuinely declining content can delay investigation of a potentially important issue.

ML is useful here because several signals can be considered together instead of relying only on one simple rule. The goal is not to build a complex automated system, but to test whether a simple and interpretable model provides useful decision-support beyond a transparent baseline.

## 2. Data safety

The analysis uses the anonymized FlyRank ML Internship dataset containing approximately 30,000 content records.

The dataset contains search-performance, content, freshness, and engagement signals.

The target was defined as:

`is_declining_label = 1` when `trend_direction == "down"`, otherwise `0`.

The final model used these five features:

* `search_volume`
* `impressions_90d`
* `days_since_last_update`
* `avg_position`
* `ctr`

Several fields were deliberately excluded.

* `trend_direction` was excluded because it directly defines the target and would cause target leakage.
* `trend_pct` was excluded because it is related to the supplied trend outcome.
* `content_id` was treated only as an identifier and was not used as a prediction feature.
* `client_id` was not used as a prediction feature. It was used only for the stricter client-grouped validation.

A temporal-overlap limitation remains because an exact future-only label window was not available. Therefore, this analysis should not be interpreted as a fully time-separated future prediction experiment.

The project was kept public-safe. No client names, private search queries, credentials, domains, or other client-identifying information are included in the project files.

## 3. Baseline

The baseline was a transparent rule-based priority score using:

* `impressions_90d`
* `days_since_last_update`

The basic idea was that content with higher visibility and greater staleness should receive higher review priority.

The baseline produced three priority categories: REFRESH, REVIEW, and MONITOR.

The baseline was used because it represents a simple approach that a content team could understand and reproduce without machine learning. This makes it a useful comparison for testing whether the ML model adds value.

On the main evaluation split, the baseline achieved:

| Metric    | Rule Baseline |
| --------- | ------------: |
| Accuracy  |         0.541 |
| Precision |         0.596 |
| Recall    |         0.475 |
| F1        |         0.529 |

The majority-class base rate was **54.21%**, so the accuracy result should be interpreted in that context.

## 4. Model / analysis

The selected model was **Logistic Regression**.

It was chosen because the project prioritizes interpretability, simplicity, and useful decision-support rather than model complexity. Logistic Regression is fast to train and allows the direction of feature relationships to be inspected.

The model used the following features:

1. `search_volume`
2. `impressions_90d`
3. `days_since_last_update`
4. `avg_position`
5. `ctr`

The target was defined as whether the supplied `trend_direction` indicated a decline.

The preprocessing pipeline used median imputation for missing numerical values and standard scaling before Logistic Regression.

The trend-related fields were intentionally excluded from the model because they would provide information too closely connected to the target.

The model was therefore designed as a simple test of whether several available search and content signals could improve prioritization over the baseline.

## 5. Evaluation

The main evaluation used an **80/20 stratified random split** with `random_state=42`.

Stratification was used to preserve the class distribution between the training and test sets.

The model and baseline were evaluated on the same test data.

| Metric    | Rule Baseline | Logistic Regression |
| --------- | ------------: | ------------------: |
| Accuracy  |         0.541 |               0.557 |
| Precision |         0.596 |               0.558 |
| Recall    |         0.475 |               0.873 |
| F1        |         0.529 |               0.681 |

The main improvement was recall. The Logistic Regression model identified substantially more of the labelled declining content, while precision decreased compared with the rule baseline.

This trade-off fits a screening use case where missing a potentially declining item may be more costly than sending some additional items for human review.

A second validation used a **client-grouped split**, keeping records from the same client together. This produced an F1 score of **0.648**, compared with **0.681** on the main random split.

The lower grouped result suggests that performance is somewhat weaker when the model must generalize across client groups. This is an important limitation and is why the results should be treated as directional decision-support rather than a production performance guarantee.

## 6. Interpretation

The model suggests that the available signals contain useful information for distinguishing content associated with the supplied decline label.

The strongest positive relationship in the fitted Logistic Regression model was associated with:

* `days_since_last_update`

This means older or less recently updated content tended to receive higher predicted decline probability in this dataset.

Among the other features, `ctr`, `avg_position`, `impressions_90d`, and `search_volume` also contributed to the model's predictions.

The important point is that these relationships are **associations in this dataset, not causal findings**.

A useful result was the strong recall achieved by the model. This supports the idea of using the model as a screening layer that helps editors find potentially risky content faster.

The main negative result was the precision trade-off. The model catches more labelled declining content, but some of the content it flags will not actually belong to the declining class.

The grouped validation result was another useful finding because it showed that performance drops when testing generalization across client groups.

## 7. Recommendation

The recommended editorial workflow is:

### 1. REFRESH

Start with the highest-priority content.

Editors should inspect whether the page is outdated, no longer matches search intent, or has lost useful information.

### 2. REVIEW

Investigate medium-priority content using additional evidence such as recent search performance, content quality, and business importance.

A model score alone should not trigger an update.

### 3. MONITOR

Keep lower-priority content under normal observation rather than spending immediate editorial resources on it.

The practical workflow is therefore:

**Rank → inspect evidence → human decision → action.**

The model should be treated as a prioritization assistant rather than an automated content-management system.

Confidence in the recommendation is **moderate**. The model showed useful measured performance and stronger recall, but the lower client-grouped F1 and the lack of a fully time-separated evaluation limit how strongly the result can be generalized.

The analysis does not claim that refreshing a flagged page will cause its search performance to improve.

## 8. Reproducibility

The complete implementation is contained in:

`work/notebooks/capstone.ipynb`

The notebook follows this general sequence:

1. Load the anonymized dataset.
2. Define the decline label.
3. Remove target-derived and identifier fields.
4. Select the five model features.
5. Build the baseline.
6. Split the data using `random_state=42`.
7. Train the Logistic Regression model.
8. Evaluate the model and baseline using the same metrics.
9. Run the client-grouped validation.
10. Generate the ranked action queue.
11. Generate the supporting analysis outputs.

The notebook is designed to run from top to bottom and contains the analysis required to reproduce the reported results.

The main evaluation uses `random_state=42` for reproducibility.

The project environment and dependencies are documented in the repository files used by the internship project.

The reported metrics should be regenerated from the notebook rather than treated as independent production benchmarks.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset.

Data and internship context credited to FlyRank.
