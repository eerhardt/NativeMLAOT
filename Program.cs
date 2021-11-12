using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using NativeMLAOT;

//Load sample data
var sampleData = new ModelInput()
{
    Age = 39F,
    Workclass = @"State-gov",
    Fnlwgt = 77516F,
    Education = @"Bachelors",
    Education_num = 13F,
    Marital_status = @"Never-married",
    Occupation = @"Adm-clerical",
    Relationship = @"Not-in-family",
    Ethnicity = @"White",
    Sex = @"Male",
    Capital_gain = 6_174F,
    Capital_loss = 0F,
    Hours_per_week = 40F,
    Native_country = @"United-States",
};

//Load model and predict output
var mlContext = new MLContext();
mlContext.ComponentCatalog.RegisterAssembly(typeof(OneHotEncodingTransformer).Assembly);
mlContext.ComponentCatalog.RegisterAssembly(typeof(FastTreeBinaryTrainer).Assembly);

string modelPath = Path.GetFullPath("MLModel1.zip");
ITransformer mlModel = mlContext.Model.Load(modelPath, out var _);

PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

ModelOutput result = predictionEngine.Predict(sampleData);

Console.WriteLine($"Predicted Label_IsOver50K_: {result.Prediction}");
Console.WriteLine($"Probability: {result.Probability}");
