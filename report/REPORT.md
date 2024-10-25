# CAST AI Summary Report & Discussion

In this document, we will discuss our solution in more detail, present summaries of findings, and discuss the results along with future improvements that can be done.

## Table of Contents

We will structure the document as follows, with discussions on the following:

1. Short Dataset Discussion
2. Findings from EDA
3. Feature Engineering Performed
4. Models Selected for Experimentation
5. Metrics Selected for Evaluation
6. Setup of Experimentation
7. Discussion of Results

As mentioned in the README.md in the root directory of the project, the Jupyter notebook `src/experiments.ipynb` is a great tool to use to follow along side this document, although we will try to summarize and cover as much as possible what was done there, and to replicate plots here for your understanding and to aid explanations.

## Short Dataset Discussion

Some notes and inferences that can be made about the dataset:

- The dataset is mainly numerical data of various kinds, various skews, and more generally speaking, various distributions. This will be important during the EDA and feature engineering steps. The data contains only a handful of categorical columns, and one datetime column.
- The dataset has a categorical column with region names commonly found in AWS as part of their region options, following the same naming conventions, e.g. eu-central-1, us-east-1, and so on. 
- The target column, labelled "label", has two values inside, namely "Interrupted" and "Continue".
- While most column names have a "_numeric" suffix to them, indicating that the column is a pure numeric column, some others have the suffix "_string", while containing float/int type values, indicating that they were either encoded or transformed at some point in some way to arrive at the numeric values.

What can we infer about this dataset then? Well, in short, this could be a dataset relating to the interruption of various spot instances on a kubernetes cluster, where we are presented with various environment variables, workload metadata, and other relevant measurements and metadata about the workload, environment, region, and so on, so that we can attempt to predict whether a spot instance will be interrupted shortly or within some fixed interval of time T shortly after the prediction. 

Assuming this 'guess' about the data origins is correct, what we have is potentially a problem with high business impact/value, which can help product teams better predict and plan-ahead for various service interruptions by either reserving more instances, or back-up or execute their workloads more efficiently with this information in mind.

### Assumptions

Some assumptions I have made while working with this dataset are:

- The rows are row-wise independent of one another - for example, I notice that there is a datetime field representing (likely) the datetime when this data was generated or collected. I assume that there is no connection between one data point and another, given that there is no ID to imply that two or more data points are connected in some way to each other. I also assume that there is no implicit time dimension to this data, in that the events do not correlate with other events in a time neighbourhood of other events, e.g. within the same minute, second, hour, etc.

### Data Preparation

We unfortunately need to do some smart sampling before we can get started: My local machine 'only' has 32GB of RAM, while Google Collab and other free platforms for ML don't offer even that much. The original data, however, leads to OOM even with batched loading. 

So, the solution is to sample it. However, we can be smart about this: If you skip ahead to the EDA on class imbalance below, you'll see that the "Interrupted" class in the data is the minority (and in the full data, by a large margin as it turns out). Therefore, I have sampled the data in the following way:

- I have loaded the data in batches
- For each batch, I take all the rows/data points with label "Interrupted", and sample down 10% of the other class

What this achieves is a kind of pseudo under-sampling of the "Continue" class, which allows me to preserve most of the signal in the "Interrupted" target class. We will again need to balance, since we still have a 1:5 class imbalance approximately, but since this is a very crude approach, I have left this to more formal methods as part of the preprocessing/feature engineering discussed below.

## Findings from the Exploratory Analysis (EDA)

### Initial Thoughts

Without a detailed data schema indicating what each of the column fields represent, and considering the high number of purely numerical columns, we are left to explore the data purely from an analytical perspective. Indeed, considering the number of fields and data, we additionally examine the columns at a macro level as well, where we look at properties such as multi-collinearity (correlations between one variable and one or more other variables), missing values present, overall skewness in distribution that can be re-normalised, and so on. 

With the above in mind, we proceed below with a cursory overview of the data, followed by more in-depth overviews of the data using our EDA helper tool, which will give us correlation matrices, a PCA analysis to help diagnose multi-collinearity via explained variance in a more 'compressed' version of the data, and more.

### Missing Values

Nothing all too fancy here. In short, if you expand the element to be scrollable, you'll find that the there are indeed many missing values in the dataset, which is indeed high as a percentage too. We'll need to address these somehow. Without putting too much thought into it, given that these are numerical fields for the overwhelming majority, we could get away with something like a median imputation, or a zero value imputation. 

Given the context of the data that we attempted to infer in the beginning, perhaps a zero-value imputation would be the most logical to use, considering that a missing value likely represents an irrelevant measurement to the behaviour being measured, or simply a metric that did not report a non-zero value at the time. Therefore, we can impute that with zero to avoid dumping the entire row.

### Feature Distributions

While we have a large number of features, we nevertheless will benefit from plotting them individually to view their distributions, in order to detect outliers, measure skewness and to determine if numerical transformations such as quantile, log, or various power transformations can help reduct the impact of outliers and to lead to a more normal distribution or uniform distribution in our final features.

To this effect, we can take a look at some plots here: see the figure below.

We don't plot all 300+ features to save one space, but the first ~50 features or so give us a good talking point as to how to choose features for transformations and to decide on which transformations we should be applying.

In short, we notice a few interesting things, namely that most of the distributions exhibit largely skewed values - skewed to the right, to be precise. Others also exhibit many outliers. And a few even exhibit multi-modal distributions. Many distributions also have a very low standard deviation around their means (i.e. are very tight), but have a very large range, illustrating that these have problematic outliers in their distributions.

How do we handle this? Well, we can transform the distributions so that they exhibit a something more similar to a normal distribution or perhaps to a uniform distribution. To do this, we can:

- Use a quantile transformation for features that take on negative values (we could also do log, after min-max scaling, but quantile is more general and faster to do in our quick experiments here), and, 
- Use a log transformation for those that are strictly positive. 

There are other techniques we can experiment with in future, namely Box-Cox and Yeo-Johnson, but we will leave those for future iterations of this project.

### Class Imbalance

From the feature distributions above, we also arrive at a feature distribution for the target classes, and we can immediately see that they are imbalanced.

With the sample data we've arrived at, the "Interrupted" class currently has about 20% of the total values of the dataset, with the remainder being "Continue".

To address this, we would need to use either under-sampling or over-sampling (or a mixture), or even synthetic example generation for the under-represented class. I'd highly advise against the latter, however, as this can cause large divergences from the real world distribution of data, and can introduce imaginary examples that will bias a model.

I would recommend under-sampling the majority class, which, while a risk of data loss for the model, doesn't risk overweighting and overfitting on specific examples from the "Interrupted" class, which can again cause the model to predict a distribution in misalignment with the real world, and can impact metrics such as precision.

### PCA Analysis and Insights

Below we see the results of a quick PCA analysis without any prior preprocessing or transformations (as discussed, we might benefit from log/quantile/power transformations). Nevertheless, we do notice some interesting behaviours:

- **The explained variance:** While somewhat expected, the explained variance quickly diminishes in increasing explained variance per added PCA component. If we set a 90% threshold on the explained variance, so as to gauge how many components we would need to roughly explain the data, we see that we would require ~100 PCA components, which is non-trivial, although it is still a 60-70% reduction in the dataset approximately, which is still a great gain in terms of computational performance, while giving us the desired multicollinearity reduction that we would like to have.
- **The first 2 PCA components:** Unfortunately don't tell us that much, since on top of large saturation of explained variance per component, we also see that first and second principle components only account for ~15-17% of the explained variance together, and so it is perhaps normal to see that they aren't entirely distinct when plotting data points.
- **The loadings of the PCA components:** These tell us the correlation of the PCA component against the raw feature of the original dataset. In short it tells us which features of the data are modelled by each PCA component, and greatly help us in being able to roughly explain what each PCA component represents in terms of derived features. However, here all our columns are anonymised and we don't have a data schema, so we can't really get to that level of explainability that we would normally be able to do. Given the large amounts of features, it would also be tedious, but if model performance justified it, and if the business use-case required such a high degree of explainability, it would likely warrant the effort. Now, what we see in our case is a reassurance that the PCA components are very distinct from one another despite each accounting for only a small portion of the explained variance. Looking out for specific groups columns from the correlation matrix that we knew were correlated to one another, we see that they are grouped into a single PCA component, which reassures us that the components are indeed orthogonal and that we likely have addressed the issue of multicollinearity in our data.

### EDA Summary of Insights

Some insights from the exploratory data analysis can be summarised here as:

- **Missing values:** There are substantial missing values in the data. I have not looked at correlations between them, but I believe there are correlations between the missing values, considering that the data (albeit without much context or schema) could pertain to different behaviours measured via the provided variables - it may show up in the correlation matrix in any case with any kind of non-zero imputation.
- **Correlations:** We do notice many clusters of variables within the 300+ raw features we have which are strongly and weakly positively correlated and some clusters of variables which are strongly and weakly negatively correlated. This could pose problems of multicollinearity, which we either need to deal with in the raw features by dropping correlated features or via PCA/TSNE/other dimensionality reduction techniques.
- **Dimensionality Reduction:** The variables themselves are quite informative, but do present themselves reasonably well to at least some dimensionality reduction (at least via PCA) it seems. Looking at the explained variance plots and the loadings, it seems we reach 90% explained variance at around 100 components, say, which is around 25-30% of the original dataset dimensionality - not bad?
- **Class imbalance:** Indeed, the class "Interrupted" is the minority class, with 20-25% of the values in the data having this label. So we need to either over-sample it, or under-sample the majority class.
- **Distribution Skewness/Shape:** Most of the distributions exhibit largely skewed values - skewed to the right, to be precise. Others also exhibit many outliers. And a few even exhibit multi-modal distributions. How do we handle this? Well, we can transform the distributions so that they exhibit a something more similar to a normal distribution or perhaps to a uniform distribution, using quantile transformations and log transformations

If we combine the introduction of quantile/log transformations with PCA, that may also introduce some interesting effects and present us with good, clean features to use for modelling.

## Feature Engineering Performed

This section naturally follows on from the EDA section above, where we make use of the insights we found about the data to engineer the features we will use for modelling. Feature engineering focused on maximizing the information provided to our models, reducing dimensionality, and managing multicollinearity between (and skewness in) the features in the data. Key features/steps are:

- **Missing Value Imputation:** We addressed missing values using zero-value imputation for numerical features, based on prior examination of the distributions and what we discussed regarding the inferred/assumed context of the data.
- **Datetime Feature Extraction:** To capture potential temporal patterns in the data, we extracted features from a `datetime64[us]` column, generating the month, day of the month, day of the week, hour, minute, and second.
- **Scaling:** Standard scaling was applied to the numerical fields to ensure consistency across features in terms of magnitude and to comply with necessities from models such as the logistic regression model and our neural network models.
- **Multicollinearity Handling (VIF and PCA):** We addressed multicollinearity using two strategies (which are chosen as part of the experimentation): VIF (Variance Inflation Factor) to detect and remove highly correlated features, or PCA (Principal Component Analysis) to retain significant variance while reducing the number of dimensions. Spoiler alert: I ended up just using PCA.
- **Transformations on Numerical Data:** Based on our analysis, we applied various transformations (Log, Quantile, and Box-Cox) to improve the interpretability and distribution of skewed numerical features. The user could specify default transformations for all numerical fields and also override individual columns with custom transformations as necessary.

### Class Imbalance

We address the class imbalance by undersampling the "Continue" class, which is the majority class. We do this ONLY for the train set, so as to improve model stability and reduce overfitting to the majority class, while leaving validation and test datasets imbalanced in order to more accurately evaluate the model as close to real-world conditions as possible.

## Models Selected for Experimentation
We chose a combination of classical machine learning models and neural networks to evaluate a range of model complexities and capabilities.

- **Logistic Regression:** Selected as a baseline, logistic regression provides a linear approach to classification. It helps establish a reference point for model performance against more complex architectures.
- **Random Forest Classifier:** As a tree-based ensemble method, random forests offer non-linear decision boundaries and are generally robust to feature correlations and multicollinearity. It captures complex patterns and interactions between features without overfitting.
- **XGBoost:** Known for its performance on structured data, XGBoost utilizes gradient-boosting techniques to optimize performance, handling imbalanced data effectively while being highly tunable.
- **Neural Network Classifiers:**
- - **Basic Neural Network:** This model features a simple architecture designed to assess the neural network’s ability to capture complex, non-linear patterns.
- - **Advanced Neural Network:** An extended version of the basic network, this model incorporates additional hidden layers, dropout, batch normalization, and various activation functions (ReLU, LeakyReLU, Tanh). It provides the model with increased representational power and regularization techniques to manage overfitting risks.

These models were chosen to balance complexity, and predictive performance.

## Metrics Selected for Evaluation

Our evaluation metrics include ROC AUC, F1 Score, Precision, and Recall. They each provide diffferent insights into model performance, especially in the context of our imbalanced dataset:

- **ROC AUC:** This metric measures the model's ability to distinguish between classes. In our context, a high ROC AUC indicates that the model effectively separates the positive and negative classes, which is essential for imbalanced datasets where the decision threshold varies.
- **F1 Score:** F1 Score combines precision and recall, making it useful for imbalanced datasets. This score is especially important for our project because it balances false positives and false negatives, providing a holistic view of performance where both types of errors have significant implications.
- **Precision:** Precision is the ratio of true positives to the sum of true positives and false positives. It’s especially valuable in contexts where false positives are costly, ensuring that positive predictions are more likely to be correct.
- **Recall:** Recall is the ratio of true positives to the sum of true positives and false negatives. This metric emphasizes minimizing false negatives, which is crucial when it’s important to capture as many positive cases as possible.

Using this combination of metrics allows us to fully understand the model's performance, and to detect risks of using these models in production, depending on the business use-cases. For example, if we have a strong requirement to catch all "Interrupted" cases with high reliability, without much concern for false positivies, we would have a high preference for near-perfect recall, with high tolerance for lower precision scores.

Conversely, if false positives are detrimental to the business's goals, then we will need to focus more on precision, at the risk of not catching some "Interrupted" cases, if we need to trade-off recall. F1 and ROC give us a more balanced view to know the overall performance of the model in the same light of false positives/false negatives

## Setup of Experimentation

The experimental setup focused on efficient, organized hyperparameter tuning and tracking across multiple models using Hyperopt for optimization, MLflow for experiment tracking, and custom classes for streamlined configuration.

### Hyperparameter Optimization with Hyperopt

We used Hyperopt for automated hyperparameter tuning across model architectures. Hyperopt allowed us to define flexible search spaces for parameters like learning rate, hidden layer sizes, dropout probabilities, and optimizer types. This facilitated efficient, targeted searches in a high-dimensional space without manually testing each configuration. We also used ROC AUC as a loss metric to the hyperopt optimiser. This can be switched to be F1, recall or precision depending on overall business use-case requirements.

### Experiment Tracking with MLflow

MLflow was employed to systematically track and log each experiment's metrics, hyperparameters, and results. Each experiment was tagged with relevant metadata (e.g., model type, feature set, and experiment version), enabling us to compare runs, visualize performance, and access artifacts in a centralized UI. Look at the README.md Installation Instructions section on how to run MLflow locally. The mlruns folder is located in the `outputs` directory, and so you should launch the MLflow UI from there

### Tags and Experiment Descriptions 

By tagging experiments with model names, feature sets, and version information, we could easily differentiate between configurations and track progress over time. Experiment descriptions provided context on specific experimental goals or configurations, assisting with future interpretations and reproducibility.

### Custom ModelOptimiser Class
The ModelOptimiser class provided a streamlined and flexible interface for managing hyperparameter optimization and model training processes. By abstracting the setup process, it handled initializing models, parameter searches, and MLflow logging, enhancing experiment reproducibility and simplifying hyperparameter tuning. The ModelOptimiser made it straightforward to define and test multiple architectures while maintaining a clean code structure.

This setup enabled efficient experimentation and fine-grained control over each model’s configuration, ensuring that we could effectively compare performance across diverse architectures and quickly iterate on results.

## Results, Discussion & Improvements

Our results indicate that feature engineering is definitely helpful. Our base logistic regression classifier performs the works, with an ROC of 0.79 approximately, and recall and precision values of <0.6. This is fairly poor for most requirements, and hence we already see we can likely do much better.

Our next set of experiments involves the full feature set, which includes feature transformations, scaling, imputation, datetime feature extraction, and 100 PCA components extracted as a final step.

Our observations:

- The best performing model in terms of ROC AUC is the XG Boost model with a value of 0.93. From there, we see that it has a fair balance between precision and recall, with approx. 0.86 and 0.90 respectively - likely achieved through its boosting that pays attention to the mistakes of the previous model in the boosting chain. If we need a balanced model that achieves good performance overall, the XG Boost with our engineered features is the way to go.
- The best performing model in terms of Recall overall is recall is the advanced neural network, which gives 0.99 recall against a fairly poor 0.50 precision (basically 1 in 2 predicted interruptions will actually be interruptions). There is another run which yields a better trade-off, with 0.95 recall against 0.64 precision (almost 2 in 3 interruptions will actually be interruptions), which is much better, again depending on the requirement for high recall, although 1 in 20 interruptions not caught at all, may be too unacceptable compared to the 1 in 100 uncaught as with the best model.

These experiments, where we use all the features together with PCA yielded the best results overall and so this is what we cover here in the discussion.

In terms of parameters and training, these models are fairly simple. The best performing random forests were those with the deepest trees (`max_depth`) and the largest number of trees (`n_estimators`), while for XG Boost a similar tendency was observed. Finally, the neural networks performed best with roughly 3-4 layers, batch normalisation, dropout and roughly 128-518 nodes (from the largest to smallest size layer) in its layers. As a whole, these models are fast and easy to train, and therefore each of them can be suitable choices for production deployment. The neural network can be further optimised for precision improvement through further model parameter tuning, training over larger epochs, and feature engineering.

Overall, we can also experiment with different loss functions for the hyperparameter optimisation, which currently optimised for ROC AUC, which may give us better results depending on the business use-case requirements for predicting interruptions correctly.


