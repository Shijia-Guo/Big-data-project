package model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.feature.ChiSqSelector;
import org.apache.spark.mllib.feature.ChiSqSelectorModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.SparkSession;

import utils.DirectoryManager;
import utils.FileManager;
import utils.LabeledPointManager;

/**
 * Feature Selection using ChiSqSelector
 * Application requires: number of output features, number of bins for discretization,dataPath, output directory path
 * 
 * Features are Double values [0,1] range. This is important for discretization
 * 
 * Feature selection tries to identify relevant features for use in model construction. 
 * It reduces the size of the feature space, which can improve both speed and statistical learning behavior.
 * 
 * ChiSqSelector implements Chi-Squared feature selection. It operates on labeled data with categorical features. 
 * ChiSqSelector uses the Chi-Squared test of independence to decide which features to choose. It supports three selection methods: numTopFeatures, percentile, fpr:
 * 
 *  -  numTopFeatures chooses a fixed number of top features according to a chi-squared test. This is akin to yielding the features with the most predictive power.
 *  -  percentile is similar to numTopFeatures but chooses a fraction of all features instead of a fixed number.
 *  -  fpr chooses all features whose p-value is below a threshold, thus controlling the false positive rate of selection.
 * 
 * By default, the selection method is numTopFeatures, with the default number of top features set to 50. The user can choose a selection method using setSelectorType.
 * The number of features to select can be tuned using a held-out validation set.
 * 
 *
 */
public class FeatureSelection_ChiSquared_Zero_One {

	@SuppressWarnings({ "serial", "resource" })
	public static void main(String[] args) throws Exception{

		//check input parameters
		if(args.length<4)
			throw new Exception("Application requires: number of output features, number of bins for discretization, dataPath, output directory path");

		int numberOfFeatures = Integer.parseInt(args[0]);
		int numberOfBinsDiscretization = Integer.parseInt(args[1]);
		String datapath = args[2];
		String outputPath = args[3];

		//for final mapping
		FileManager ioManager = new FileManager();
		BufferedWriter out = null;
		BufferedWriter statistics = null;

		//check output directory
		DirectoryManager.checkAndDelete(Paths.get(outputPath));
		if(!Files.exists(Paths.get(outputPath))){
			new File(outputPath).mkdir();
		}

		long startTime = System.currentTimeMillis();

		//Properties set directly on the SparkConf take highest precedence, then flags passed to spark-submit or spark-shell, then options in the spark-defaults.conf file.
		SparkSession spark;
		spark = SparkSession.builder()
				//.master("local[4]")
				.appName("FeatureSelection_ChiSquared")
				.getOrCreate();
		JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
		JavaRDD<String> rawInputRdd = jsc.textFile(datapath);

		//extract labeles points
		JavaRDD<LabeledPoint> points = LabeledPointManager.prepareLabeledPoints(rawInputRdd, null);

		//===============================================================================================================================
		out = new BufferedWriter(new FileWriter(outputPath+"/discretization.txt"));
		ioManager.addLine(String.valueOf(points.take(1).get(0).features().size()), out);
		ioManager.addLine(points.take(1).toString(), out);

		// Discretize data in equal bins since ChiSqSelector requires categorical features
		// ChiSqSelector treats each unique value as a category
		final double divider = 100./ numberOfBinsDiscretization;
		JavaRDD<LabeledPoint> discretizedData = points.map(
				new Function<LabeledPoint, LabeledPoint>() {
					@Override
					public LabeledPoint call(LabeledPoint lp) {
						final double[] discretizedFeatures = new double[lp.features().size()];
						for (int i = 0; i < lp.features().size(); ++i) {
							discretizedFeatures[i] = Math.floor(lp.features().apply(i) * 100 / divider);
						}
						return new LabeledPoint(lp.label(), Vectors.dense(discretizedFeatures));
					}
				}
				);
		ioManager.addLine(String.valueOf(discretizedData.take(1).get(0).features().size()), out);
		ioManager.addLine(discretizedData.take(1).toString(), out);
		out.close();

		//===============================================================================================================================
		// Create ChiSqSelector 
		ChiSqSelector selector = new ChiSqSelector(numberOfFeatures);
		final ChiSqSelectorModel transformer = selector.fit(discretizedData.rdd());

		//WRITE EXTRACTED FEATURES
		out = new BufferedWriter(new FileWriter(outputPath+"/features.csv"));
		for(int l: transformer.selectedFeatures())
			ioManager.addLine("feature "+l, out);
		out.close();	


		//FLUSH STATISTICS
		long endTime = System.currentTimeMillis();
		ioManager.writeString("OVERALL TIME " + (endTime - startTime)/1000 + " seconds", outputPath+"/statistics.txt");	

		jsc.stop();
		spark.stop();
	}

}
