> Formulation of different baselines. For example, u → i  denotes that these models aim to predict target i for user u. Item i can be represented by itemID in IDNet, pre-extracted features (in ViNet), or an image encoder in PixelNet.

Model | Formulation | InputType
--- | --- | ---
MF, VBPR, LightGCN | u→i   | InputType.Pair 
DSSM, FM |  i1, i2 ...in−1 → in; i1, i2 ...in → in−1; ... i2, i3 ...in → i1; | InputType.SEQ
ACF |  u, i1, i2 ...in−1 → in; u, i1, i2 ...in → in−1 ; ... u, i2, i3 ...in → i1; | InputType.SEQ
GRU4Rec, NextItNet, SASRec |  i1, i2 ...in−1 → i2, i3 ...in; | InputType.SEQ
BERT4Rec | i1, [MASK], ...in → i2 | InputType.SEQ
SRGNN, LightSANs | i1→ i2; i1, i2→ i3; ... i1, i2 ...in−1 → in; | InputType.AUGSEQ
VisRank |   i1, i2 ...iNu−1 → iNu | InputType.SEQ/ Pair 






> Introduction of the code pipeline

We supply a brief guide on implementing new models based on the pipeline. The foremost step is determining the model type. We've divided models into three fundamental types within the pipeline: IDNet models, which only model ID features; ViNet models which utilize pre-extracted visual features for recommendation; and PixelNet models, which train image encoders end-to-end with recommendation tasks. The implementation details for these three models vary, hence we discuss them individually:

1. Building traditional recommendation models (IDNet).
   The implementation steps are akin to those in *RecBole*, with four functions requiring instantiation:
   
   - `_init_()`: This function is used for network structure construction, load and definition of global variables, parameters initialization, etc.
   
   - `forward()`: This function is employed for the optimization process of the model, calculating the forward propagation loss for a batch of training data.
   
   - `compute_item_all()`: This function is used to calculate the entire item representations, primarily used for the model evaluation process.

   - `predict()`: This function is used in the model evaluation process, generating the input user's scores for the entire item pool.


2. Building traditional visual recommendation models (ViNet).
   The construction process is the same with IDNet, with the addition that we offer `load_weight` function to facilitate processing visual feature vectors in such models. It is straightforward to include new application methods for visual features by extending the `load_weight` function.

3. Building end-to-end training visual recommendation models (PixelNet).
   We offer the `load_model` function to aid in loading and applying image encoders in PixelNet models. In such models, given the GPU memory limitations, it's typically not feasible to compute all item representations at once. Therefore, different from the previous construction process, the model needs to implement the third function as follows:

   - `compute_item()`: This function is used to compute the representations of an input batch of items, primarily used in the model evaluation process.
   

The steps outlined above ensure the correct definition of a new model in the pipeline. In the end, it is necessary to choose the input and output data formats for the new model to perform model training and evaluation with reliance on pipeline interfaces. Multiple implementations of data formats are provided in `data.dataset` module, see the upper table for reference. In the model, specifying the class variable `input_type` to set a particular data format. For example, setting `input_type == InputType.Pair` corresponds to the data format of the 1st row from Table. Then with the usage of `data.utils` function, you can bind the model name with corresponding train, valid, and test dataset names, thus completing the process of model input and output data format finalization.   


> Thanks to the excellent code repository [RecBole](https://github.com/RUCAIBox/RecBole) and [VisRec](https://github.com/ialab-puc/VisualRecSys-Tutorial-IUI2021) ! 







