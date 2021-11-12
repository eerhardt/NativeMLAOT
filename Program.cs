using Microsoft.ML;
using Microsoft.ML.Transforms;
using NativeMLAOT;

// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");


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
mlContext.ComponentCatalog.RegisterAssembly(typeof(Microsoft.ML.Trainers.FastTree.FastTreeBinaryTrainer).Assembly);

string MLNetModelPath = Path.GetFullPath("MLModel1.zip");
ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

var result = predictionEngine.Predict(sampleData);

Console.WriteLine($"Predicted Label_IsOver50K_: {result.Prediction}");
Console.WriteLine($"Probability: {result.Probability}");
