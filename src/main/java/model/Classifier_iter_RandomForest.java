package model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.SparkSession;

import scala.Tuple2;
import utils.DirectoryManager;
import utils.ErrorComputer;
import utils.FileManager;
import utils.LabeledPointManager;
import utils.PathManager;
import utils.TranslateRFModel;

public class Classifier_iter_RandomForest {
	@SuppressWarnings({ "serial", "resource" })
	public static void main( String[] args ) throws Exception
	{
		//********************************************** LOADING **********************************************
		//check input parameters
		if(args.length<7)
			throw new Exception("Application requires: maxNumberOfIterations, minFMeasure, maxDepth, maxBins, number of trees, dataPath, output directory path");

		//stopping conditions
		Integer maxIterations = Integer.parseInt(args[0]);
		Double minFMeasure = Double.parseDouble(args[1]);
		
		//output dir must be local
		String outputDir = args[6];
		if(outputDir.startsWith("hdfs") || outputDir.startsWith("local"))
			throw new Exception("Outpur directory is local by default. You cannot specify the prefix for the path");

		//support files/directories
		String datapath = args[5];
		//check output directory
		DirectoryManager.checkAndDelete(Paths.get(outputDir));
		if(!Files.exists(Paths.get(outputDir))){
			new File(outputDir).mkdir();
		}

		//prepare writing output
		FileManager ioManager = new FileManager();

		//no support for HDFS. Output is local
		BufferedWriter statistics = new BufferedWriter(new FileWriter(outputDir+"/statistics"));

		//check if there are features to ignore
		Set<Integer> featuresToIgnore = new HashSet<Integer>();

		//random forest settings
		Integer numClasses = 2;
		HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		Integer numTrees = Integer.parseInt(args[4]); 
		String featureSubsetStrategy = "auto";
		String impurity = "gini";
		Integer maxDepth = Integer.parseInt(args[2]);//only supported <=30
		Integer maxBins = Integer.parseInt(args[3]);
		Integer seed = 5121985;

		//stopping conditions
		int iter = 0;
		double fMeasureCurr = 1.0;

		//********************************************** SPARK **********************************************
		SparkSession spark;	
		spark = SparkSession.builder()
				.appName("Classifier_Iterat_RandomForest")
				.getOrCreate();
		JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

		// Load and parse the data file.	
		long startTime = System.currentTimeMillis();
		JavaRDD<String> rawInputRdd = jsc.textFile(datapath);

		

		//iterate
		long date = new Date().getTime();
		while(fMeasureCurr>=minFMeasure && iter<maxIterations) {
			try {
				long iterStartTime = System.currentTimeMillis();
				//manage directory
				String outputDirIteration = outputDir+"/"+iter;
				DirectoryManager.checkAndDelete(Paths.get(outputDirIteration));
				if(!Files.exists(Paths.get(outputDirIteration))){
					new File(outputDirIteration).mkdir();
				}

				ioManager.addLine("=======================================================", statistics);
				ioManager.addLine("=== ITERATION "+iter+" ===", statistics);

				//extract labeles points
				JavaRDD<LabeledPoint> parsedData = LabeledPointManager.prepareLabeledPoints(rawInputRdd, featuresToIgnore);

				// Split the data into training and test sets (30% held out for testing)
				JavaRDD<LabeledPoint>[] splits = parsedData.randomSplit(new double[]{0.7, 0.3});
				JavaRDD<LabeledPoint> trainingData = splits[0];
				JavaRDD<LabeledPoint> testData = splits[1];

				// Train a RandomForest model.
				final RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses,
						categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
						seed);

				long endBuildTime = System.currentTimeMillis();
				ioManager.addLine("BUILDING model: " + (endBuildTime - iterStartTime)/1000 + " seconds", statistics);

				//write the trees
				String forestName = outputDirIteration+"/forest.txt";
				ioManager.writeString(model.toDebugString(), forestName);

				// Evaluate model on test instances and compute test error
				JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
				ErrorComputer errorComp = new ErrorComputer();
				Double testErr = errorComp.fMeasure(predictionAndLabel);
				long endEvalTime = System.currentTimeMillis();
				ioManager.addLine("F-Measure on test data: " + String.format("%.2f", testErr*100)+"%", statistics);
				ioManager.addLine("EVALUATION of the model (test data): " + (endEvalTime - endBuildTime)/1000 + " seconds", statistics);
				predictionAndLabel.unpersist();

				//update stopping conditions
				fMeasureCurr = testErr;

				//evaluate the model on training data
				predictionAndLabel = trainingData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
				testErr = errorComp.fMeasure(predictionAndLabel);
				endEvalTime = System.currentTimeMillis();
				ioManager.addLine("F-Measure on training data: " + String.format("%.2f", testErr*100)+"%", statistics);
				ioManager.addLine("EVALUATION of the model (training data): " + (endEvalTime - endBuildTime)/1000 + " seconds", statistics);

				// Save the model: method default is an HDFS path, but application default is file
				try{
					String targetModel = PathManager.getInstance().checkPathWithDefault(outputDirIteration+"/RF_"+date, "file");
					model.save(jsc.sc(), targetModel);			
				}
				catch(Exception e){
					e.printStackTrace();
				}

				long endTime = System.currentTimeMillis();
				ioManager.addLine("ITERATION TIME " + (endTime - iterStartTime)/1000 + " seconds", statistics);

				//extract features and add to the list to be removed
				TranslateRFModel.extractCpgFromForest(forestName, outputDirIteration, false);
				featuresToIgnore.addAll(ioManager.parseFeatureNumbers(outputDirIteration+"/features.txt"));

				//release memory
				predictionAndLabel.unpersist(true);
				trainingData.unpersist(true);
				testData.unpersist(true);
				parsedData.unpersist(true);
				System.out.println("Memory Released");
			}
			catch(Exception e){
				e.printStackTrace();
			}
			finally {
				iter++;
			}
		}

		long endTime = System.currentTimeMillis();
		ioManager.addLine("\nOVERALL TIME " + (endTime - startTime)/1000 + " seconds", statistics);
		statistics.close();

		//write all extracted features
		//no support for HDFS. Output is local
		BufferedWriter out = new BufferedWriter(new FileWriter(outputDir+"/allFeatures.csv"));
		for(Integer i: featuresToIgnore)
			ioManager.addLine("feature "+i, out);
		out.close();

		jsc.stop();
		spark.stop();
	}
}
