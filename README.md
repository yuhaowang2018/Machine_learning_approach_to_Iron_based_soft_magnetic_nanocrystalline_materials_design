# Machine learning approach to Fe-based soft magnetic nanocrystalline materials design

## Introduction 

As  applied  in  a  great  number  of  research  areas,  machine  learning  is  currently  playing  a significant role in materials design.  In this work, we utilized machine learning techniques to efficiently boost the development of soft magnetic materials.  This process includes building a database composed of published experimental results, utilizing machine learning methods on  it,  thus  identifying  the  trends  of  magnetic  properties  in  soft  magnetic  materials,  and accelerating the design of next-generation soft magnetic nanocrystalline materials through the  use  of  numerical  optimization.   Machine  learning  regression  models  were  trained  to predict magnetic saturation, coercivity and magnetostriction and further to use a stochastic optimization framework to optimize the corresponding magnetic properties.

To  verify  the  feasibility  of  the  machine  learning  model,  several  optimized  soft  magnetic materials—specified in terms of composition and thermomechanical treatments—have been predicted and then prepared and tested, which shows great consistency between predictions and experiments, proving the reliability of the design model.

## Dependencies

- Pandas >1.0.0
- Numpy >1.13
- Matplotlib > 3.2
- Scikit-learn > 0.22
- Scipy > 1.2

## Contents

Different folders inside the repository are designated for different parts of our machine learning approach:

### Data visualizations

The folder contains two python notebooks `Data Visualizations Learning.ipynb` and `Database Statistics.ipynb`  that display an overview of all the data we collected from literatures. The total number of entries of data we collected is 1440. 

The data curation procedure follows the steps below:

1. Remove all data which is missing annealing temperature, annealing time, all as-quenched data, and all data processed below room temperature
2. Round Annealing Temperatures to typical processing values - every 5th degree Celsius
3. Round Annealing Times to typical processing values - nearest half hour
4. Remove points out of nanocrystalline regime - grain diameter over 60nm
6. Remove any features which are unused after data reduction

We removed 146 entries after curation process.

### Feature selection

The folder contains a python notebook `Select Features.ipynb` which uses the following five methods to identify features for removal:

1. Find columns with a missing percentage greater than 50%
2. Find columns with a single unique value
3. Find collinear variables with a correlation greater than 95%
4. Find features with 0.0 feature importance from a gradient boosting machine (GBM)
5. Find features that contribute less than 95% to a specified cumulative feature importance from the GBM

### Machine learning model 

We use `Orange` software for preliminary screening of a suitable machine learning model. The file `MachineLearningWorkflow.ows` is the main file that shows our work flow in Orange to select the final model from a range of different model choices. The python notebook `LearningResults.ipynb` shows the performance comparison between five different algorithms and the predicted capability of random forest algorithm. 

The machine learning model training includes the following procedures: 

1. Prepare the data based on feature space after previous feature selection (dimensionality reduction) process and separate.
2. Utilize five different algorithms: Random Forest, kNN, Decision Tree,  SVM and Linear Regression.
3. When evaluating the performance, we use 20-fold cross validation procedure and replace the null values in features with the mean response.

As results show, Random forest performs better in most properties. 

### First iteration optimization

The python notebook `First iteration optimization.ipynb` contains the procedure we used for our first iteration of optimization.  In this iteration, we attempted to maximize magnetic saturation while keeping coercivity and magnetostriction as low as possible. 

The optimization steps include:

1. Merge features of magnetic saturation, magnetostriction and coercivity together, fill zeros in null values.
2. Build random forest model on each properties separately.
3. Impose constraints on composition space, the sum of all the elements cannot exceed 100.
4. Utilize differential evolution algorithm with four different optimization strategy:
   - Constrain ln(coercivity) < -1.5; constrain magnetostriction < 3; maximizing magnetic saturation
   - Constrain magnetostriction < 3 ; minimizing (-magnetic saturation+coercivity)
   - Constrain ln(coercivity) < -0.5; constrain magnetostriction < 3; maximizing magnetic saturation.
   - Constrain magnetostriction < 3 ; minimizing (-magnetic saturation*4+ ln(coercivity) )
5. Combine results from all the strategy and pick the most suitable choices. 

### Second iteration optimization

The python notebook `Second iteration of optimization.ipynb` describes the procedure we used for our second iteration of optimization after the experimental validation of the first iteration. In this iteration, we narrowed our focus to maximize magnetic saturation and minimize coercivity. In addition, the composition space is constrained to be in the FINEMET range. 

The optimization steps include:

1. Merge available features of coercivity and magnetic saturation together, fill zeros in null values.. 
2. Build random forest model on coercivity and magnetic saturation separately. 
3. Impose constraints on composition space, the sum of all the elements cannot exceed 100.
4. Utilize differential evolution algorithm with the following optimization strategy:
   - Constrain ln(coercivity)<0.5; maximizing magnetic saturation
5. The optimization has four different constraints on element selection:
   - Include all the elements of interest.
   - Constrain one element of the group "Ge, Mo, Nb, P" to zero.
   - Constrain two elements of the group "Ge, Mo, Nb, P" to zero.
   - Constrain three elements of the group "Ge, Mo, Nb, P" to zero.
6. Combine results from all the different constraints and pick the most suitable choices. 



