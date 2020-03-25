# Machine learning approach to Fe-based soft magnetic nanocrystalline materials design

#### Introduction

As  applied  in  a  great  number  of  research  areas,  machine  learning  is  currently  playing  a significant role in materials design.  In this work, we utilized machine learning techniques to efficiently boost the development of soft magnetic materials.  This process includes building a database composed of published experimental results, utilizing machine learning methods on  it,  thus  identifying  the  trends  of  magnetic  properties  in  soft  magnetic  materials,  and accelerating the design of next-generation soft magnetic nanocrystalline materials through the  use  of  numerical  optimization.   Machine  learning  regression  models  were  trained  to predict magnetic saturation, coercivity and magnetostriction and further to use a stochastic optimization framework to optimize the corresponding magnetic properties.

To  verify  the  feasibility  of  the  machine  learning  model,  several  optimized  soft  magnetic materials—specified in terms of composition and thermomechanical treatments—have been predicted and then prepared and tested, which shows great consistency between predictions and experiments, proving the reliability of the design model.

#### Contents 

Different folders inside the repository are designated for different parts of our machine learning approach:

##### Data Visualizations

The folder contains two python notebooks that display an overview of all the data we collected from literatures. 

##### Feature selection

The folder contains a python notebook which uses the following five methods to identify features for removal:

    1. Find columns with a missing percentage greater than 50%
    2. Find columns with a single unique value
    3. Find collinear variables with a correlation greater than 95%
    4. Find features with 0.0 feature importance from a gradient boosting machine (gbm)
    5. Find features that contribute less than 95% to a specified cumulative feature     	    importance from the gbm
##### Machine learning model 

We use "Orange" software for preliminary screening of a suitable machine learning model. The file "MachineLearningWorkflow.ows" is the main file that shows our work flow in Orange to select the final model from a range of different model choices. 