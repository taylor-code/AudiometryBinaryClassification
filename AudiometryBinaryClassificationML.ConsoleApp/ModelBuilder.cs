using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using AudiometryBinaryClassificationML.Model;

namespace AudiometryBinaryClassificationML.ConsoleApp
{
    public static class ModelBuilder
    {
        /* FILE PATHS */
        //private static string APP_PATH => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string APP_PATH => Environment.CurrentDirectory;
        /// <summary> Dataset used to train the model. </summary>
        /*private static string _trainDataPath => Path.Combine(APP_PATH, "..", "..", "..", "Data", "AudiometryTrain.csv");
        /// <summary> Dataset used to evaluate the model. </summary>
        private static string _testDataPath => Path.Combine(APP_PATH, "..", "..", "..", "Data", "AudiometryTest.csv");
        /// <summary> Dataset where the trained model is saved. </summary>
        private static string _modelPath => Path.Combine(APP_PATH, "..", "..", "..", "Models", "model.zip");*/
        /// <summary> Dataset where the trained model is saved. </summary>
        private static string TEST_DATA_FILEPATH = Path.Combine(APP_PATH, "..", "..", "..", "Data", "AudiometryTrain.csv");
        /// <summary> Dataset used to train the model. </summary>
        private static string TRAIN_DATA_FILEPATH = Path.Combine(APP_PATH, "..", "..", "..", "Data", "AudiometryTest.csv");
        /// <summary> Dataset where the trained model is saved. </summary>
        private static string MODEL_FILE = ConsumeModel.MLNetModelPath;


        /// <summary> Provides processing context. </summary>
        private static MLContext mlContext;

        private static ITransformer TrainedModel;



        /// <summary>
        /// Driver for the model creation.
        /// </summary>
        public static void CreateModel()
        {
            // All ML.NET operations start in the MLContext class.
            mlContext = new MLContext(seed: 1);
            Console.WriteLine("Initialized MLContext.");

            // Load data.
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: TRAIN_DATA_FILEPATH,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);
            Console.WriteLine("Loaded the training data.");

            // Build training pipeline.
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext);
            Console.WriteLine("Processed the data.");

            // Train model.
            //ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline);
            TrainedModel = TrainModel(mlContext, trainingDataView, trainingPipeline);
            Console.WriteLine("Trained the model.");

            // Evaluate model.
            Evaluate(mlContext, trainingDataView, trainingPipeline);

            // Save model.
            SaveModel(mlContext, TrainedModel, MODEL_FILE, trainingDataView.Schema);
        }


        /// <summary>
        /// Extracts and transforms the data.
        /// </summary>
        /// <returns> trainingPipeline </returns>
        public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            // Extract features and transform the data.
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("col0", "col0")
                                      .Append(mlContext.Transforms.Categorical.OneHotEncoding(new[] {
                                          new InputOutputColumnPair("col1", "col1"), new InputOutputColumnPair("col2", "col2"),
                                          new InputOutputColumnPair("col3", "col3"), new InputOutputColumnPair("col4", "col4"),
                                          new InputOutputColumnPair("col5", "col5"), new InputOutputColumnPair("col6", "col6"),
                                          new InputOutputColumnPair("col7", "col7"), new InputOutputColumnPair("col8", "col8"),
                                          new InputOutputColumnPair("col9", "col9"), new InputOutputColumnPair("col10", "col10"),
                                          new InputOutputColumnPair("col11", "col11"), new InputOutputColumnPair("col12", "col12"),
                                          new InputOutputColumnPair("col13", "col13") }
                                      ))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("col14_tf", "col14"))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("col15_tf", "col15"))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("col16_tf", "col16"))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("col17_tf", "col17"))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("col18_tf", "col18"))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("col19_tf", "col19"))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("col20_tf", "col20"))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("col21_tf", "col21"))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("col22_tf", "col22"))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("col23_tf", "col23"))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("col24_tf", "col24"))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("col25_tf", "col25"))
                                      .Append(mlContext.Transforms.Concatenate("Features", new[] { "col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10", "col11", "col12", "col13", "col14_tf", "col15_tf", "col16_tf", "col17_tf", "col18_tf", "col19_tf", "col20_tf", "col21_tf", "col22_tf", "col23_tf", "col24_tf", "col25_tf" }));
            
            // Set the training algorithm.
            var trainer = mlContext.MulticlassClassification.Trainers.LightGbm(labelColumnName: @"col0", featureColumnName: "Features")
                                      .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }


        /// <summary>
        /// Fits the training data to the training pipeline.
        /// </summary>
        /// <returns> model </returns>
        public static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
             ITransformer model = trainingPipeline.Fit(trainingDataView);
            return model;
        } 


        /// <summary>
        /// Cross-Validates a single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
        /// </summary>
        /// <returns> model </returns>
        private static void Evaluate(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            var testDataView = mlContext.Data.LoadFromTextFile<ModelInput>(TEST_DATA_FILEPATH, hasHeader: true);

            // Gets the quality metrics for the model.
            //var testMetrics = mlContext.MulticlassClassification.Evaluate(TrainedModel.Transform(testDataView));

            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "col0");
            PrintMulticlassClassificationFoldsAverageMetrics(crossValidationResults);
        }


        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));
        }


        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }


        public static void PrintMulticlassClassificationMetrics(MulticlassClassificationMetrics metrics)
        {
            /*
             * Metrics for Multi-Class Classification:
             * 
             * Micro Accuracy      -  Better if close to 1
             * Macro Accuracy      -  Better if close to 1
             * Log-loss            -  Better if close to 0
             * Log-loss Reduction  -  Better if close to 1
             */

            // Display the metrics for model validation.
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for Multi-Class Classification Model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    MacroAccuracy = {metrics.MacroAccuracy:0.####}");
            Console.WriteLine($"    MicroAccuracy = {metrics.MicroAccuracy:0.####}");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}");

            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                Console.WriteLine($"    LogLoss for class {i + 1} = {metrics.PerClassLogLoss[i]:0.####}");
            }

            Console.WriteLine($"************************************************************");
        }


        public static void PrintMulticlassClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValResults)
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
            var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);

            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
            var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);

            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
            var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);

            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
            var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            Console.WriteLine($"*************************************************************************************************************");

        }


        /// <summary>
        /// Calculates the standard deviation.
        /// </summary>
        /// <param name="values"></param>
        /// <returns>The standard deviation.</returns>
        public static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();

            // Calculate the standard deviation.
            return Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
        }


        /// <summary>
        /// Calculates the confidence interval of 95.
        /// </summary>
        /// <param name="values"></param>
        /// <returns>The confidence interval.</returns>
        public static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            return 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
        }
    }
}
