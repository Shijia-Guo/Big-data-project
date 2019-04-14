package model;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Date;
import java.util.HashMap;

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

/**
 * RandomForest classifier
 * Application requires: maxDepth, maxBins, number of trees, dataPath, output directory path, (optional) file path with list of features to ignore
 * 
 * INPUT: CSV file (the header is skipped). The first column is skipped, the last column is the category, features are comma separated
 * Features are Double values in any range
 * 
 *
 */
public class Classifier_RandomForest {

	@SuppressWarnings({ "serial", "resource" })
	public static void main( String[] args ) throws Exception
	{

		//check input parameters
		if(args.length<5)
			throw new Exception("Application requires: maxDepth, maxBins, number of trees, inputFile, output directory path");

		//Properties set directly on the SparkConf take highest precedence, then flags passed to spark-submit or spark-shell, then options in the spark-defaults.conf file.
		SparkSession spark;
		//add a master if it is in java CL options, otherwise set it in spark-submit command
		spark = SparkSession.builder()
				.appName("Classifier_RandomForest")
				.getOrCreate();
		JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

		String datapath = args[3];
		String outputDir = args[4];

		//check output directory
		DirectoryManager.checkAndDelete(Paths.get(outputDir));
		if(!Files.exists(Paths.get(outputDir))){
			new File(outputDir).mkdir();
		}

		//prepare writing output
		FileManager ioManager = new FileManager();
		BufferedWriter statistics = new BufferedWriter(new FileWriter(outputDir+"/statistics"));


		// Load and parse the data file.	
		long startTime = System.currentTimeMillis();
		JavaRDD<String> rawInputRdd = jsc.textFile(datapath);
		//extract labeles points
		JavaRDD<LabeledPoint> parsedData = LabeledPointManager.prepareLabeledPoints(rawInputRdd,null);

		// Split the data into training and test sets (30% held out for testing)
		JavaRDD<LabeledPoint>[] splits = parsedData.randomSplit(new double[]{0.7, 0.3});
		JavaRDD<LabeledPoint> trainingData = splits[0];
		JavaRDD<LabeledPoint> testData = splits[1];

		// Train a RandomForest model.
		Integer numClasses = 2;
		HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<>(); //all features are continuous.
		Integer numTrees = Integer.parseInt(args[2]); // Use more in practice.
		String featureSubsetStrategy = "auto"; // Let the algorithm choose.
		String impurity = "gini";
		Integer maxDepth = Integer.parseInt(args[0]);//only supported <=30
		Integer maxBins = Integer.parseInt(args[1]);
		Integer seed = 5121985;

		final RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses,
				categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
				seed);

		long endBuildTime = System.currentTimeMillis();
		ioManager.addLine("BUILDING model: " + (endBuildTime - startTime)/1000 + " seconds", statistics);

		//write the trees
		String treeName = outputDir+"/forest.txt";
		ioManager.writeString(model.toDebugString(), treeName);

		// Evaluate model on test instances and compute test error
		JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
		ErrorComputer errorComp = new ErrorComputer();
		Double testErr = errorComp.fMeasure(predictionAndLabel);
		long endEvalTime = System.currentTimeMillis();
		ioManager.addLine("F-Measure on test data: " + String.format("%.2f", testErr*100)+"%", statistics);
		ioManager.addLine("EVALUATION of the model (test data): " + (endEvalTime - endBuildTime)/1000 + " seconds", statistics);

		//evaluate the model on training data
		predictionAndLabel = trainingData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
		testErr = errorComp.fMeasure(predictionAndLabel);
		endEvalTime = System.currentTimeMillis();
		ioManager.addLine("F-Measure on training data: " + String.format("%.2f", testErr*100)+"%", statistics);
		ioManager.addLine("EVALUATION of the model (training data): " + (endEvalTime - endBuildTime)/1000 + " seconds", statistics);

		// Save the model
		long date = new Date().getTime();
		try{
			String targetModel = PathManager.getInstance().checkPathWithDefault(outputDir+"/RF_"+date, "file");
			model.save(jsc.sc(), targetModel);
		}
		catch(Exception e){
			e.printStackTrace();
		}

		long endTime = System.currentTimeMillis();
		ioManager.addLine("OVERALL TIME " + (endTime - startTime)/1000 + " seconds", statistics);

		//extract features named after MLlib naming convention
		TranslateRFModel.extractCpgFromForest(treeName, outputDir, false);

		jsc.stop();
		spark.stop();
	}

}
