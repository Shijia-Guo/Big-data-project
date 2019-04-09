package utils;

import java.util.Map;
import java.util.TreeMap;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.regression.LabeledPoint;

public class LabeledPointManager {
	
	/**
	 * Parse the date without header to extract labeledPoints
	 * The last column is the category
	 * The first column is the ID of the observation, to be removed
	 * There can be null values, denoted by ?. We use sparse vectors then.
	 * 
	 * @param dataNoHeader
	 * @param featureToBeIgnored can be null
	 * @param featureToInclude can be null
	 * @return
	 */
	@SuppressWarnings({ "serial" })
	public static JavaRDD<LabeledPoint> prepareLabeledPoints(JavaRDD<String> rawData) {
		return rawData.map(new Function<String, LabeledPoint>() {
			public LabeledPoint call(String line) throws Exception {
				String[] parts = line.split("\\s+");
				Map<Integer, Double> index2value = new TreeMap<Integer, Double>();
				for(int i=0;i<parts.length-1;i++){
						if(!parts[i].equals("-1"))
							index2value.put(i, Double.parseDouble(parts[i]));
				}
				int size = parts.length-1;
				int[] indices = new int[index2value.size()];
				double[] values = new double[index2value.size()];
				int i = 0;
				for(int index: index2value.keySet()){
					indices[i] = index;
					values[i] = index2value.get(index);
					i++;
				}
				double dLabel = Double.parseDouble(parts[size]);
				return new LabeledPoint(dLabel,	new SparseVector(size, indices, values));
			}
		});
	}

	


}
